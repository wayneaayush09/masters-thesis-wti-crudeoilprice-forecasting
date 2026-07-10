# ============================================================
# longhorizon_v2/_lh_sota_models.py
# Compact (B,T,C)->(B,n_outputs) implementations of recent long-horizon SOTA
# models, REGISTERED INTO THE IN-MEMORY MODEL_REGISTRY (02_model_shared.py is
# NOT edited). They accept build_model's standard kwargs (fall into its
# `else: Cls(**common)` branch), so train_band_model(model_name="TiDE") and
# run_standalone(model_name="TiDE") work immediately.
#   • DLinearNet / NLinearNet  (Zeng et al. 2023 — linear beats transformers)
#   • TiDE                       (Das et al. 2023 — dense MLP encoder-decoder)
#   • FITS                       (Xu et al. 2024 — frequency-domain linear)
#   • SegRNN                     (Lin et al. 2023 — segment GRU)
# ============================================================

if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())

import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _te:
    raise RuntimeError(f"_lh_sota_models requires torch: {_te}. Run _env.py first.") from _te


class DLinearNet(nn.Module):
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, kernel=25, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        self.kernel = min(kernel, lookback if lookback % 2 == 1 else lookback - 1)
        self.lin_trend = nn.Linear(lookback, n_outputs)
        self.lin_seas  = nn.Linear(lookback, n_outputs)
    def _ma(self, xt):                              # xt: (B,T)
        k = self.kernel
        pad = k // 2
        x = F.pad(xt.unsqueeze(1), (pad, pad), mode="replicate")
        t = F.avg_pool1d(x, k, stride=1).squeeze(1)
        return t[:, :xt.size(1)]
    def forward(self, x):                           # (B,T,C)
        x = self.revin(x); xt = x.mean(dim=-1)       # (B,T)
        trend = self._ma(xt); seas = xt - trend
        return self.lin_trend(trend) + self.lin_seas(seas)   # (B,n_outputs)


class NLinearNet(nn.Module):
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        self.lin = nn.Linear(lookback, n_outputs)
    def forward(self, x):
        x = self.revin(x); xt = x.mean(dim=-1)       # (B,T)
        last = xt[:, -1:]
        return self.lin(xt - last) + last            # broadcast (B,1) -> (B,n_outputs)


class _ResMLP(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))


class TiDE(nn.Module):
    """Time-series Dense Encoder: flatten lookback -> residual-MLP encoder -> decoder."""
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        din = lookback * n_features
        self.proj = nn.Linear(din, d_model)
        self.enc = nn.ModuleList([_ResMLP(d_model, dropout) for _ in range(max(n_layers, 2))])
        self.dec = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, n_outputs))
        # global residual: linear lookback->n_outputs on the channel mean (NLinear-style skip)
        self.skip = nn.Linear(lookback, n_outputs)
    def forward(self, x):                            # (B,T,C)
        x = self.revin(x); B = x.size(0)
        h = self.proj(x.reshape(B, -1))
        for blk in self.enc: h = blk(h)
        return self.dec(h) + self.skip(x.mean(dim=-1))


class FITS(nn.Module):
    """Frequency Interpolation Time Series: rfft -> low-mode complex linear -> head."""
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, modes=32, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        self.modes = max(1, min(modes, lookback // 2))
        self.cw = nn.Parameter(0.02 * torch.randn(self.modes, self.modes, dtype=torch.cfloat))
        self.head = nn.Sequential(nn.Linear(2 * self.modes, d_ff), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_ff, n_outputs))
    def forward(self, x):                            # (B,T,C)
        x = self.revin(x); xt = x.mean(dim=-1)        # (B,T)
        Xf = torch.fft.rfft(xt, dim=1)                # (B, T//2+1)
        m = min(self.modes, Xf.size(1))
        z = Xf[:, :m]                                 # (B,m)
        z = torch.einsum("bm,mn->bn", z, self.cw[:m, :m])
        feat = torch.cat([z.real, z.imag], dim=1)     # (B,2m)
        if feat.size(1) < 2 * self.modes:             # pad if T tiny
            feat = F.pad(feat, (0, 2 * self.modes - feat.size(1)))
        return self.head(feat)


class SegRNN(nn.Module):
    """Segment RNN: split lookback into segments, GRU over segments, decode."""
    def __init__(self, n_features, lookback, d_model=64, n_layers=2, d_ff=256,
                 n_outputs=1, dropout=0.2, n_seg=None, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        seg = n_seg or max(2, lookback // 12)
        while lookback % seg != 0 and seg > 1:
            seg -= 1
        self.seg_len = lookback // seg
        self.n_seg = seg
        self.gru = nn.GRU(self.seg_len, d_model, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_ff, n_outputs))
    def forward(self, x):                            # (B,T,C)
        x = self.revin(x); B = x.size(0)
        xt = x.mean(dim=-1)[:, :self.n_seg * self.seg_len]
        xs = xt.reshape(B, self.n_seg, self.seg_len)  # (B, n_seg, seg_len)
        out, hn = self.gru(xs)
        return self.head(hn[-1])                       # (B,n_outputs)


MODEL_REGISTRY.update({
    "DLinearNet": DLinearNet, "NLinearNet": NLinearNet,
    "TiDE": TiDE, "FITS": FITS, "SegRNN": SegRNN,
})
print("✅ longhorizon_v2/_lh_sota_models.py loaded — registered:",
      ["DLinearNet", "NLinearNet", "TiDE", "FITS", "SegRNN"])
