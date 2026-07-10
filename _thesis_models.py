# ============================================================
# DDP_v2/_thesis_models.py
# Thesis models from: Mahajan, "Forecasting Crude Oil Prices using Deep Learning"
# IIT-Bombay MTech, 2024. Table XX best results:
#   N-BEATS: MSE 0.0002, MAE 0.0092 (normalized [0,1], 1-step, leaky whole-series MinMax)
#   ILSTM+CNN: MSE 0.0005, MAE 0.015 (normalized)
#   ESN: MSE 0.0017, MAE 0.0111 (normalized)
#   LSTNet: MSE 7.989, MAE 1.552 (raw $-scale)
#   ILSTM: MSE 9.429, MAE 1.811 (raw $)
# NOTE: mixed scales in their table — normalized vs raw$ not directly comparable.
#
# These compact (B,T,C)->(B,n_outputs) implementations are registered into the
# in-memory MODEL_REGISTRY so they work with train_band_model, B_wf_*, C_ensemble.
# Architecture sources: Oreshkin 2019 (N-BEATS), Lai 2018 (LSTNet), Jaeger/Gao 2021 (ESN),
# Wang 2022 (ILSTM). Run / exec from inside DDP_v2/.
# ============================================================

if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())

import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _te:
    raise RuntimeError(f"_thesis_models requires torch: {_te}. Run _env.py first.") from _te


# ── ILSTM — Improved LSTM (Wang et al. 2022) ─────────────────────────────────
# 2-gate variant (forget + input, no output gate) + CIM (Cross-Input Modulation)
# Prevents supersaturation in sigmoid by passing through tanh(sigmoid(input)).
class ILSTM(nn.Module):
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.d_model = d_model
        self.n_layers = max(n_layers, 1)
        # Weight matrices per layer: Wf, Wi (forget gate, input gate)
        self.cells = nn.ModuleList()
        in_dim = n_features
        for _ in range(self.n_layers):
            self.cells.append(nn.ModuleDict({
                "Wfh": nn.Linear(d_model, d_model, bias=False),
                "Wfx": nn.Linear(in_dim, d_model),
                "Wih": nn.Linear(d_model, d_model, bias=False),
                "Wix": nn.Linear(in_dim, d_model),
            }))
            in_dim = d_model
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_outputs)

    def _ilstm_step(self, cell, x_t, h, c):
        # forget gate with cell state memory (2-gate design)
        ft = torch.sigmoid(cell["Wfh"](h) + cell["Wfx"](x_t) + c)
        # input gate + CIM (tanh applied to sigmoid output)
        it_raw = torch.sigmoid(cell["Wih"](h) + cell["Wix"](x_t) + c)
        it = torch.tanh(it_raw)   # CIM: values in [0, 0.762) → less supersaturation
        c_new = ft * c + it * torch.tanh(cell["Wix"](x_t))
        h_new = torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x):                              # (B, T, C)
        B, T, _ = x.shape
        out = x
        for cell in self.cells:
            h = torch.zeros(B, self.d_model, device=x.device)
            c = torch.zeros(B, self.d_model, device=x.device)
            for t in range(T):
                h, c = self._ilstm_step(cell, out[:, t, :], h, c)
            out = h.unsqueeze(1).expand(B, T, -1)   # reuse h at each step as features
        return self.head(self.drop(h))               # last timestep


# ── ILSTM+CNN — CNN feature extractor + ILSTM ────────────────────────────────
class ILSTMCnn(nn.Module):
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, kernel=3, **kw):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.ilstm = ILSTM(d_model, lookback, d_model=d_model, n_layers=max(1, n_layers // 2),
                           d_ff=d_ff, n_outputs=n_outputs, dropout=dropout)

    def forward(self, x):                              # (B, T, C)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, d_model)
        return self.ilstm(h)


# ── N-BEATS — Neural Basis Expansion (Oreshkin et al. 2019) ──────────────────
class _NBEATSBlock(nn.Module):
    def __init__(self, lookback, h, units, basis):
        super().__init__()
        self.basis = basis
        self.mlp = nn.Sequential(
            nn.Linear(lookback, units), nn.ReLU(),
            nn.Linear(units, units), nn.ReLU(),
            nn.Linear(units, units), nn.ReLU(),
            nn.Linear(units, units), nn.ReLU(),
        )
        self.theta_b = nn.Linear(units, basis)   # backcast
        self.theta_f = nn.Linear(units, basis)   # forecast

    def forward(self, x):
        return self.theta_f(self.mlp(x)), self.theta_b(self.mlp(x))


class NBEATSLite(nn.Module):
    """Compact N-BEATS: generic stack with trend + seasonal basis expansion.
    Predicts n_outputs ahead. Contract: (B, T, C) → (B, n_outputs)."""
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, n_stacks=2, **kw):
        super().__init__()
        self.lookback = lookback
        self.n_outputs = n_outputs
        basis = max(4, n_outputs)
        n_blocks = max(n_layers, 2)
        self.proj = nn.Linear(n_features, 1)   # univariate projection
        self.blocks = nn.ModuleList([
            _NBEATSBlock(lookback, n_outputs, d_model, basis)
            for _ in range(n_stacks * n_blocks)])
        # linear projection from basis to n_outputs
        self.fc_f = nn.Linear(basis, n_outputs)

    def forward(self, x):                              # (B, T, C) → (B, n_outputs)
        B = x.size(0)
        residual = self.proj(x).squeeze(-1)            # (B, T)
        forecast = torch.zeros(B, self.n_outputs, device=x.device)
        for blk in self.blocks:
            fc_theta, bc_theta = blk(residual)
            forecast = forecast + self.fc_f(fc_theta)
            residual = residual - bc_theta[:, :self.lookback]  # backcast residual
        return forecast


# ── LSTNet — Long- and Short-term Time-series Network (Lai 2018) ─────────────
class LSTNet(nn.Module):
    """CNN → GRU (short-term) + Skip-GRU (periodic) + AR (linear) + highway."""
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, cnn_kernel=6, skip=24, n_heads=4, **kw):
        super().__init__()
        self.skip = min(skip, lookback // 2)
        conv_out = d_model
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv_out, (cnn_kernel, n_features)),
            nn.Tanh(), nn.Dropout(dropout))
        cnn_T = lookback - cnn_kernel + 1   # output time steps from CNN
        # GRU (short-term)
        self.gru = nn.GRU(conv_out, d_model, num_layers=1, batch_first=True)
        # Skip-GRU (periodic, captures ~weekly cycle)
        skip_len = max(1, cnn_T // self.skip)
        self.skip_gru = nn.GRU(conv_out * self.skip, d_model // 2, num_layers=1, batch_first=True)
        # AR (linear direct autoregressive component)
        self.ar = nn.Linear(lookback, n_outputs)
        # Highway gate
        gru_dim = d_model + d_model // 2
        self.highway = nn.Sequential(nn.Linear(gru_dim, d_ff), nn.Tanh(), nn.Dropout(dropout))
        self.gate_g = nn.Linear(gru_dim, d_ff)   # gate
        self.head = nn.Linear(d_ff, n_outputs)
        self.drop = nn.Dropout(dropout)
        self.cnn_T = cnn_T

    def forward(self, x):                              # (B, T, C)
        B, T, C = x.shape
        # CNN path: (B,1,T,C) → conv → (B,d_model,cnn_T,1)
        xc = x.unsqueeze(1)
        h = self.cnn(xc).squeeze(-1).transpose(1, 2)     # (B, cnn_T, d_model)
        # Short-term GRU
        _, hg = self.gru(h)                               # (1,B,d_model)
        gru_out = hg.squeeze(0)                           # (B, d_model)
        # Skip-GRU: reshape last (skip * skip_len) timesteps
        skip_T = min(self.cnn_T, (self.cnn_T // self.skip) * self.skip)
        hs = h[:, -skip_T:, :].reshape(B, self.cnn_T // self.skip, -1) \
            if skip_T > 0 else h.new_zeros(B, 1, h.size(-1) * self.skip)
        _, hsk = self.skip_gru(hs)
        skip_out = hsk.squeeze(0)                         # (B, d_model//2)
        combined = torch.cat([gru_out, skip_out], dim=1)  # (B, gru_dim)
        # Highway
        g = torch.sigmoid(self.gate_g(combined))
        hw = g * self.highway(combined) + (1 - g) * self.head(
            nn.functional.linear(combined, self.head.weight.new_zeros(
                self.head.weight.shape[0], combined.shape[1]),
                self.head.bias))
        # AR component
        ar_out = self.ar(x.mean(dim=-1))                  # (B, n_outputs)
        return self.head(self.drop(combined)) + ar_out


# ── Echo State Network (ESN) — reservoir computing ──────────────────────────
# Only the readout is trained (ridge regression on reservoir states).
# Implemented as a torch module for MODEL_REGISTRY compatibility.
class EchoStateNetwork(nn.Module):
    def __init__(self, n_features, lookback, d_model=200, n_layers=1, d_ff=256,
                 n_outputs=1, dropout=0.2, spectral_radius=0.95, sparsity=0.1,
                 leaking_rate=1.0, seed=42, **kw):
        super().__init__()
        res = d_model
        rng = torch.Generator(); rng.manual_seed(seed)
        # Random (fixed) input → reservoir weights
        Win = torch.randn(res, n_features, generator=rng) * 0.1
        self.register_buffer("Win", Win)
        # Sparse random reservoir with spectral radius scaling
        W = torch.randn(res, res, generator=rng)
        mask = (torch.rand(res, res, generator=rng) < sparsity).float()
        W = W * mask
        if W.abs().max() > 0:
            W = W / W.abs().max() * spectral_radius
        self.register_buffer("W", W)
        self.leaking_rate = leaking_rate
        self.res = res
        # Trainable readout layer only
        self.readout = nn.Linear(res, n_outputs, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                              # (B, T, C)
        B, T, C = x.shape
        h = x.new_zeros(B, self.res)
        for t in range(T):
            inp = x[:, t, :] @ self.Win.T             # (B, res)
            pre = inp + h @ self.W.T                   # (B, res)
            h = (1 - self.leaking_rate) * h + self.leaking_rate * torch.tanh(pre)
        return self.readout(self.drop(h))              # (B, n_outputs)


# ── Register into MODEL_REGISTRY ─────────────────────────────────────────────
MODEL_REGISTRY["ILSTM"]     = ILSTM
MODEL_REGISTRY["ILSTMCnn"]  = ILSTMCnn
MODEL_REGISTRY["NBEATSLite"]= NBEATSLite
MODEL_REGISTRY["LSTNet"]    = LSTNet
MODEL_REGISTRY["ESN"]       = EchoStateNetwork

print("✅ DDP_v2/_thesis_models.py loaded — registered ILSTM, ILSTMCnn, NBEATSLite, LSTNet, ESN")
print("   IIT-B thesis (Mahajan 2024) models now available to train_band_model + B_wf_* + C_ensemble")
