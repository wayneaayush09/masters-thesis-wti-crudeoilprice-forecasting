# ============================================================
# DDP_v2/_fedformer_plus.py  — FEDformer deep-dive (extra-focus)
# Adds two compact (B,T,C)->(B,n_outputs) frequency architectures to the
# in-memory MODEL_REGISTRY (02_model_shared.py is NOT edited):
#   • FEDformerW   — wavelet / multi-resolution enhanced block (complement to the
#                    Fourier FEDformerLite already in 02_model_shared.py).
#   • FreEformer   — 2025 frequency-enhanced transformer (compact reimpl): a learnable
#                    full-spectrum enhancement + attention over frequency bins.
# Both reuse RevIN + _make_head + _SeriesDecompMulti from 02_model_shared.py and fall
# into build_model's `else: Cls(**common)` branch, so train_band_model(model_name=...)
# and run_paper_* work unchanged. Run from inside DDP_v2/.
# ============================================================
if "MODEL_REGISTRY" not in globals():
    exec(open("02_model_shared.py", encoding="utf-8").read())

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _te:
    raise RuntimeError(f"_fedformer_plus requires torch: {_te}. Run _env.py first.") from _te


# ── FEDformer-w : multi-resolution (wavelet-like) enhanced block ─────────────
class _MultiResBlock(nn.Module):
    """Pool the sequence at scales [1,2,4,8], linear-transform each scale in d_model
    space, upsample back to T and sum — a compact stand-in for the wavelet enhanced
    block (multi-resolution analysis without DWT bookkeeping; shape-safe)."""
    def __init__(self, d_model, lookback, scales=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.scales = [s for s in scales if s <= max(1, lookback)]
        self.lins = nn.ModuleList([nn.Linear(d_model, d_model) for _ in self.scales])
        self.drop = nn.Dropout(dropout)
    def forward(self, x):                              # (B,T,d_model)
        B, T, D = x.shape
        out = torch.zeros_like(x)
        xt = x.transpose(1, 2)                          # (B,D,T)
        for lin, s in zip(self.lins, self.scales):
            if s == 1:
                out = out + lin(x)
            else:
                pooled = F.avg_pool1d(xt, kernel_size=s, stride=s, ceil_mode=True)  # (B,D,T')
                up = F.interpolate(pooled, size=T, mode="linear", align_corners=False)
                out = out + lin(up.transpose(1, 2))
        return self.drop(out)


class FEDformerW(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        self.proj  = nn.Linear(n_features, d_model)
        self.decomp = _SeriesDecompMulti(d_model)       # MOEDecomp (from 02_model_shared)
        nb = max(n_layers, 2)
        self.blocks = nn.ModuleList([_MultiResBlock(d_model, lookback, dropout=dropout) for _ in range(nb)])
        self.norm   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nb)])
        self.ffn    = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)) for _ in range(nb)])
        self.head   = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x); h = self.proj(x)
        seasonal, trend = self.decomp(h)
        s = seasonal
        for blk, nrm, ffn in zip(self.blocks, self.norm, self.ffn):
            s = nrm(s + blk(s)); s = nrm(s + ffn(s))
        return self.head((s + trend).mean(dim=1))


# ── FreEformer : frequency-enhanced transformer (compact 2025 reimpl) ────────
class FreEformer(nn.Module):
    """Embed → rFFT over time → learnable full-spectrum enhancement (per-bin complex
    scaling) + a small Transformer attending over frequency bins (real⊕imag) →
    iFFT → head. Uses the WHOLE spectrum (not a fixed low-mode cut) which is the
    FreEformer distinction from FEDformer's mode selection."""
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, n_heads=4, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        self.proj  = nn.Linear(n_features, d_model)
        self.nbins = lookback // 2 + 1
        # learnable full-spectrum enhancement (real + imag gains per bin, per channel)
        self.enh_r = nn.Parameter(torch.ones(self.nbins, d_model))
        self.enh_i = nn.Parameter(torch.zeros(self.nbins, d_model))
        nh = n_heads if (2 * d_model) % n_heads == 0 else 4
        enc = nn.TransformerEncoderLayer(d_model=2 * d_model, nhead=nh, dim_feedforward=d_ff,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.freq_enc = nn.TransformerEncoder(enc, num_layers=max(1, n_layers // 2))
        self.head = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x); h = self.proj(x)             # (B,T,d_model)
        T = h.size(1)
        Xf = torch.fft.rfft(h, dim=1)                   # (B,F,d_model) complex
        F_ = Xf.size(1)
        er = self.enh_r[:F_].unsqueeze(0); ei = self.enh_i[:F_].unsqueeze(0)
        Xf = Xf * torch.complex(er, ei)                 # full-spectrum enhancement
        z = torch.cat([Xf.real, Xf.imag], dim=-1)       # (B,F,2*d_model) real
        z = self.freq_enc(z)                            # attention over frequency bins
        d = z.size(-1) // 2
        Xf2 = torch.complex(z[..., :d], z[..., d:])     # back to complex (B,F,d_model)
        rec = torch.fft.irfft(Xf2, n=T, dim=1)          # (B,T,d_model)
        return self.head(rec.mean(dim=1))


MODEL_REGISTRY["FEDformerW"] = FEDformerW
MODEL_REGISTRY["FreEformer"] = FreEformer

# ── improved per-horizon config knobs for the FEDformer family ───────────────
# Consumed by _models_plus.py / train_band_model via train_kwargs (d_model etc.).
FEDFORMER_CONFIG = {
    "1d":   {"d_model": 64,  "n_layers": 2, "modes": 16},
    "7d":   {"d_model": 64,  "n_layers": 2, "modes": 24},
    "90d":  {"d_model": 128, "n_layers": 3, "modes": 48},
    "180d": {"d_model": 128, "n_layers": 3, "modes": 64},
}

print("✅ DDP_v2/_fedformer_plus.py loaded — registered FEDformerW, FreEformer")
