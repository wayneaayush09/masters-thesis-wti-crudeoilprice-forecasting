# ============================================================
# FILE 02 — Shared PatchTST Architecture, Dataset & Training
# Aligned with "A Time Series is Worth 64 Words" (Nie et al., 2023)
# Includes RevIN + AMP mixed-precision training.
# NOT run directly. Each forecast file does:
#   exec(open('02_model_shared.py', encoding="utf-8").read())
# ============================================================
import math, random, gc
from copy import deepcopy
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except Exception as _te:
    TORCH_AVAILABLE = False
    raise RuntimeError(
        f"02_model_shared: torch is required for neural-network models but failed to load: {_te}\n"
        "Run _env.py first (it pre-loads torch once) and restart the kernel if the error persists."
    ) from _te
from sklearn.preprocessing import RobustScaler

# ── Constants (paper-aligned) ────────────────────────────────
BATCH_SIZE  = 128       # paper default
MAX_EPOCHS  = 100       # paper-style longer training
LR          = 1e-4      # paper default
PATIENCE    = 15
QUANTILES   = [0.05, 0.2, 0.5, 0.8, 0.95]

# ── Device detection: CUDA GPU → TPU (XLA) → CPU ─────────────
# TPU v5e in Colab needs torch_xla. Install with:
#   !pip install -q torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
IS_CUDA = torch.cuda.is_available()

# ── Metrics-neutral speed flags (CUDA only) ──────────────────
# cudnn.benchmark autotunes conv kernels for the fixed per-horizon input shapes
# (lookback × n_features fixed; only batch count varies). Biggest win for the
# conv models (TCN, TCNBiLSTM, CNNBiLSTMAttn). TF32 is free on Ampere/Ada GPUs
# (L4/A100) and a no-op on T4. None of these change the math beyond ~1e-6 (TF32
# ~1e-3 on Ampere+), well below reported MAE precision.
if IS_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Speed profile (one switch: env var DDP_SPEED) ────────────
# Trades a small, DISCLOSED accuracy shift for big wall-clock savings without
# touching code. Applied here so every driver that exec's this file inherits it.
#   'full'       → paper defaults (100/15/128, LR 1e-4) — UNCHANGED behaviour
#   'balanced'   → recommended Phase-2 setting (50 epochs, patience 8, batch 256)
#   'aggressive' → also raises LR (largest shift; validate vs old numbers first)
# Combine with N_SEEDS=2 + WF_FREQ_MONTHS in the run cell for the full speed-up.
import os as _os_speed
_DDP_SPEED = _os_speed.environ.get("DDP_SPEED", "full").lower()
if _DDP_SPEED == "balanced":
    MAX_EPOCHS = 50
    PATIENCE   = 8
    BATCH_SIZE = 256 if IS_CUDA else BATCH_SIZE
elif _DDP_SPEED == "aggressive":
    MAX_EPOCHS = 35
    PATIENCE   = 6
    BATCH_SIZE = 256 if IS_CUDA else BATCH_SIZE
    LR         = 3e-4
if _DDP_SPEED in ("balanced", "aggressive"):
    print(f"⏩ DDP_SPEED='{_DDP_SPEED}'  →  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}  "
          f"BATCH_SIZE={BATCH_SIZE}" + (f"  LR={LR}" if _DDP_SPEED == 'aggressive' else ""))

def _get_device():
    if IS_CUDA:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        print(f"✅ TPU detected: {dev}")
        return dev
    except ImportError:
        print("⚠️  No GPU/TPU — running on CPU")
        return torch.device("cpu")

DEVICE = _get_device()
IS_TPU = (not IS_CUDA) and ("xla" in str(DEVICE).lower())

def _free_memory():
    """Device-agnostic memory cleanup."""
    if IS_CUDA:
        torch.cuda.empty_cache()
    elif IS_TPU:
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()   # flush pending XLA computation graph
        except ImportError:
            pass
    gc.collect()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── RevIN (Reversible Instance Normalization) ────────────────
# Key PatchTST component: per-instance normalize before encoder,
# stabilizes training across regime changes (e.g. $20 vs $100 oil).
# We apply NORM only (target is a log-return scalar, not a feature value,
# so no denorm needed for our scalar regression).

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps          = eps
        self.affine       = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        # x: (B, T, C)
        mean = x.mean(dim=1, keepdim=True)
        std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x

# ── PatchTST Architecture ─────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, n_features, d_model, dropout=0.2):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        n_patches = T // self.patch_len
        x = x[:, :n_patches*self.patch_len, :]
        x = x.reshape(B, n_patches, self.patch_len*C)
        return self.drop(self.norm(self.proj(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class PatchTST(nn.Module):
    def __init__(self, n_features, patch_len, lookback, d_model,
                 n_heads, n_layers, d_ff, n_outputs=1, dropout=0.2):
        super().__init__()
        self.revin     = RevIN(n_features)
        self.patch_emb = PatchEmbedding(patch_len, n_features, d_model, dropout)
        self.pos_enc   = PositionalEncoding(d_model, max_len=lookback//patch_len+8, dropout=dropout)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head      = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff//2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff//2, n_outputs))
    def forward(self, x):
        x = self.revin(x)                    # ← PAPER: per-instance norm
        x = self.patch_emb(x); x = self.pos_enc(x)
        x = self.encoder(x);   x = x.mean(dim=1)
        return self.head(x)

# ============================================================
# COMPARATOR ARCHITECTURES (for the multi-model benchmark)
# All share PatchTST's contract: input (B, T, C) → output (B, n_outputs).
# Each reuses RevIN (per-instance norm) and a shared regression head so the
# downstream pipeline (Student-t / MSE / quantile heads, naive-shrinkage,
# split-conformal CI) works identically across every model.
# These are COMPACT reimplementations for honest benchmarking — NOT the
# official thuml / NeuralForecast checkpoints.
# ============================================================

def _make_head(d_in, d_ff, n_outputs, dropout):
    return nn.Sequential(
        nn.LayerNorm(d_in), nn.Linear(d_in, d_ff // 2),
        nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff // 2, n_outputs))

# ── TCN — dilated causal temporal conv network (Bai et al. 2018) ──
class _Chomp(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous() if self.chomp else x

class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.GELU(), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.GELU(), nn.Dropout(dropout))
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return nn.functional.gelu(out + res)

class TCN(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, kernel=3, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        chans = [d_model] * max(n_layers, 2)
        layers, c_in = [], n_features
        for i, c_out in enumerate(chans):
            layers.append(_TCNBlock(c_in, c_out, kernel, 2**i, dropout)); c_in = c_out
        self.tcn  = nn.Sequential(*layers)
        self.head = _make_head(chans[-1], d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x).transpose(1, 2)        # (B,C,T)
        x = self.tcn(x)[:, :, -1]                # last timestep features
        return self.head(x)

# ── TCN-BiLSTM — TCN encoder feeding a bidirectional LSTM ─────────
class TCNBiLSTM(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, kernel=3, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        chans = [d_model] * max(n_layers, 2)
        layers, c_in = [], n_features
        for i, c_out in enumerate(chans):
            layers.append(_TCNBlock(c_in, c_out, kernel, 2**i, dropout)); c_in = c_out
        self.tcn  = nn.Sequential(*layers)
        self.lstm = nn.LSTM(chans[-1], d_model // 2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x).transpose(1, 2)        # (B,C,T)
        x = self.tcn(x).transpose(1, 2)          # (B,T,C)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])          # last step of BiLSTM

# ── TimeMixer-lite — past-decomposable MLP mixing (Wang et al. 2024) ──
class _SeriesDecomp(nn.Module):
    """Moving-average trend / seasonal split."""
    def __init__(self, kernel=25):
        super().__init__(); self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2)
    def forward(self, x):                         # x: (B,T,C)
        t = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if t.size(1) != x.size(1): t = t[:, :x.size(1), :]
        return x - t, t                           # seasonal, trend

class TimeMixer(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin   = RevIN(n_features)
        self.decomp  = _SeriesDecomp(25)
        self.proj    = nn.Linear(n_features, d_model)
        # time-mixing (across T): LayerNorm applied BEFORE transpose so it
        # normalises the d_model dim (last dim of h:(B,T,d_model)).
        # Previous version put LayerNorm inside the Sequential after transpose,
        # where last dim = T ≠ d_model → crash when T ≠ d_model (e.g. lookback=30, d_model=64).
        self.time_mix = nn.ModuleList([
            nn.Sequential(nn.Linear(lookback, lookback), nn.GELU(), nn.Dropout(dropout))
            for _ in range(max(n_layers, 2))])
        self.time_mix_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(max(n_layers, 2))])
        # feature-mixing (across d_model): unchanged — last dim is already d_model
        self.feat_mix = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout)) for _ in range(max(n_layers, 2))])
        self.head = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x)
        seasonal, trend = self.decomp(x)
        h = self.proj(seasonal + trend)           # (B,T,d_model)
        for i, (tm, fm) in enumerate(zip(self.time_mix, self.feat_mix)):
            h_norm = self.time_mix_norm[i](h)                         # LayerNorm on d_model
            h = h + tm(h_norm.transpose(1, 2)).transpose(1, 2)        # mix across time
            h = h + fm(h)                                              # mix across features
        return self.head(h.mean(dim=1))

# ── N-HiTS-lite — multi-rate pooling + MLP stacks (Challu et al. 2023) ──
class _NHiTSBlock(nn.Module):
    def __init__(self, lookback, n_features, d_model, pool, dropout):
        super().__init__()
        self.pool = nn.MaxPool1d(pool, stride=pool, ceil_mode=True)
        pooled_T  = math.ceil(lookback / pool)
        self.mlp  = nn.Sequential(
            nn.Linear(pooled_T * n_features, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU())
    def forward(self, x):                         # x: (B,T,C)
        B = x.size(0)
        p = self.pool(x.transpose(1, 2)).transpose(1, 2)     # (B,T',C)
        return self.mlp(p.reshape(B, -1))

class NHiTS(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin  = RevIN(n_features)
        pools = [1, 2, 4, 8][:max(n_layers, 2)]
        self.blocks = nn.ModuleList(
            [_NHiTSBlock(lookback, n_features, d_model, p, dropout) for p in pools])
        self.head = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x)
        h = sum(b(x) for b in self.blocks)        # hierarchical residual sum
        return self.head(h)

# ── Informer-lite — ProbSparse-style attention + distilling (Zhou 2021) ──
class _ProbSparseAttn(nn.Module):
    """Approximate ProbSparse: score queries by max-mean gap, keep top-u."""
    def __init__(self, d_model, n_heads, dropout, factor=5):
        super().__init__()
        self.h = n_heads; self.dk = d_model // n_heads; self.factor = factor
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        u = min(T, int(self.factor * math.ceil(math.log(max(T, 2)))))
        sparsity = scores.max(-1)[0] - scores.mean(-1)        # (B,h,T) query importance
        idx = sparsity.topk(u, dim=-1)[1]                     # top-u queries
        mask = torch.zeros(B, self.h, T, device=x.device, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn) * mask.unsqueeze(-1)           # zero out non-selected queries
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.o(out)

class Informer(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_heads, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        self.proj  = nn.Linear(n_features, d_model)
        self.pos   = PositionalEncoding(d_model, max_len=lookback + 8, dropout=dropout)
        self.attn  = nn.ModuleList([_ProbSparseAttn(d_model, n_heads, dropout)
                                    for _ in range(max(n_layers, 2))])
        self.norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(max(n_layers, 2))])
        self.ffn   = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)) for _ in range(max(n_layers, 2))])
        # distilling conv halves the time dimension between layers
        self.distil = nn.ModuleList([nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
                                     for _ in range(max(n_layers, 2) - 1)])
        self.head  = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x); h = self.pos(self.proj(x))
        for i, (a, n, f) in enumerate(zip(self.attn, self.norm, self.ffn)):
            h = n(h + a(h)); h = n(h + f(h))
            if i < len(self.distil):
                h = self.distil[i](h.transpose(1, 2)).transpose(1, 2)
        return self.head(h.mean(dim=1))

# ── Autoformer-lite — series decomp + Auto-Correlation (Wu 2021) ──
class _AutoCorrelation(nn.Module):
    """FFT-based period aggregation approximating Auto-Correlation attention."""
    def __init__(self, d_model, n_heads, dropout, factor=3):
        super().__init__()
        self.h = n_heads; self.dk = d_model // n_heads; self.factor = factor
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).permute(0, 2, 3, 1)   # (B,h,dk,T)
        k = self.k(x).view(B, T, self.h, self.dk).permute(0, 2, 3, 1)
        v = self.v(x).view(B, T, self.h, self.dk).permute(0, 2, 3, 1)
        # autocorrelation via FFT along time — cast to float32: cuFFT in fp16
        # requires power-of-2 sequence lengths which our lookbacks (60/96/336/360) are not.
        _fft_dtype = q.dtype
        qf = torch.fft.rfft(q.float(), dim=-1); kf = torch.fft.rfft(k.float(), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=T, dim=-1).to(_fft_dtype)  # (B,h,dk,T)
        corr = corr.mean(dim=2)                                          # (B,h,T) lag scores
        k_top = min(T, int(self.factor * math.ceil(math.log(max(T, 2)))))
        w, delay = torch.topk(corr, k_top, dim=-1)                       # (B,h,k_top) weights+lags
        w = torch.softmax(w, dim=-1)
        # Vectorised time-delay aggregation. The previous version rolled v with a
        # Python `for b in range(B)` loop + `int(shift[b].float().mean())`, forcing a
        # GPU→CPU sync per batch element per lag — the Autoformer slowdown. Here each
        # top lag rolls v along T via gather over the whole (B,h) batch at once (no
        # host sync). Per-head lags are now honoured individually (closer to the true
        # Auto-Correlation aggregation than the old mean-over-heads scalar).
        base = torch.arange(T, device=x.device)                          # (T,)
        agg  = torch.zeros_like(v)                                        # (B,h,dk,T)
        for i in range(k_top):
            d   = delay[:, :, i]                                          # (B,h) integer lags
            idx = (base.view(1, 1, T) - d.unsqueeze(-1)) % T             # (B,h,T) rolled positions
            idx = idx.unsqueeze(2).expand(B, self.h, self.dk, T)         # (B,h,dk,T)
            rolled = torch.gather(v, 3, idx)                             # roll along T per (B,h)
            agg = agg + rolled * w[:, :, i].unsqueeze(-1).unsqueeze(-1)
        out = agg.permute(0, 3, 1, 2).reshape(B, T, -1)
        return self.o(self.drop(out))

class Autoformer(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_heads, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin  = RevIN(n_features)
        self.proj   = nn.Linear(n_features, d_model)
        self.decomp = _SeriesDecomp(25)
        self.ac     = nn.ModuleList([_AutoCorrelation(d_model, n_heads, dropout)
                                     for _ in range(max(n_layers, 2))])
        self.ffn    = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)) for _ in range(max(n_layers, 2))])
        self.head   = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x); h = self.proj(x)
        for a, f in zip(self.ac, self.ffn):
            s, _ = self.decomp(h + a(h)); h = s                # keep seasonal after AC
            s2, _ = self.decomp(h + f(h)); h = s2              # keep seasonal after FFN
        return self.head(h.mean(dim=1))

# ── iTransformer — inverted dims, attention ACROSS variables (Liu 2024) ──
# Embeds each variable's WHOLE lookback series as one token, then runs
# multi-head attention over the n_features tokens to model cross-variable
# interactions (interest rates × supply × VIX × oil). Opposite philosophy
# to PatchTST's channel-independence. Direct WTI precedent (review ref 86).
class iTransformer(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_heads, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, **kw):
        super().__init__()
        self.revin    = RevIN(n_features)
        self.var_embed = nn.Linear(lookback, d_model)         # each variable series → token
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x)                 # (B,T,C)
        x = x.transpose(1, 2)             # (B,C,T) → treat each variable as a token
        tokens = self.var_embed(x)        # (B,C,d_model): one token per variable
        enc = self.encoder(tokens)        # attention across the C variable-tokens
        return self.head(enc.mean(dim=1)) # pool variable tokens → head

# ── TFT (Temporal Fusion Transformer) with quantile head (Lim 2021) ──────────
# Variable-selection network (VSN) + local LSTM + multi-head attention.
# Direct WTI precedent: He et al. Procedia 2023; Zha et al. Procedia 2024.
# When loss="quantile", the quantile head directly emits calibrated intervals.
# VSN gate also provides interpretable feature-importance for the thesis.
class TFT(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, n_heads=4, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        # Variable Selection Network: gates each input feature
        self.vsn_gate = nn.Sequential(
            nn.Linear(n_features, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_features), nn.Sigmoid())
        self.input_proj = nn.Linear(n_features, d_model)
        # Local LSTM: captures short-range temporal patterns
        lstm_layers = max(n_layers // 2, 1)
        self.lstm = nn.LSTM(d_model, d_model // 2,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        # Temporal self-attention: captures long-range dependencies
        attn_layers = max(n_layers // 2, 1)
        n_h = n_heads if d_model % n_heads == 0 else (4 if d_model % 4 == 0 else 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_h, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=attn_layers)
        self.head = _make_head(d_model, d_ff, n_outputs, dropout)

    def forward(self, x):
        x = self.revin(x)                               # (B, T, C)
        gate = self.vsn_gate(x.mean(dim=1, keepdim=True))  # (B, 1, C)
        x = x * gate                                    # feature selection
        h = self.input_proj(x)                          # (B, T, d_model)
        h, _ = self.lstm(h)                             # BiLSTM: (B, T, d_model)
        h = self.attn(h)                                # temporal attention
        return self.head(h.mean(dim=1))

# ── CNN-BiLSTM with Multi-head Attention (C2) ───────────────────────────────
# Direct WTI precedent: Foroutan & Lahmiri Financial Innovation 2024 (#2 behind TCN);
# An et al. VMD-PSR-CNN-BiLSTM Eng. Applic. AI 2024.
# CNN extracts local features → BiLSTM models temporal dependencies →
# self-attention re-weights timesteps → regression head.
class CNNBiLSTMAttn(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, n_heads=4, **kw):
        super().__init__()
        self.revin = RevIN(n_features)
        half = max(d_model // 2, 16)
        # Two-layer CNN feature extractor (causal via padding)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, half, kernel_size=3, padding=1),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(half, d_model, kernel_size=3, padding=1),
            nn.GELU(), nn.Dropout(dropout))
        # BiLSTM for temporal context
        lstm_layers = max(n_layers // 2, 1)
        self.lstm = nn.LSTM(d_model, d_model // 2,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        # Self-attention re-weights all timesteps
        n_h = n_heads if d_model % n_heads == 0 else (4 if d_model % 4 == 0 else 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_h, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = _make_head(d_model, d_ff, n_outputs, dropout)

    def forward(self, x):
        x = self.revin(x)                         # (B, T, C)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, d_model)
        h, _ = self.lstm(h)                       # (B, T, d_model)
        h = self.attn(h)                          # (B, T, d_model)
        return self.head(h.mean(dim=1))

# ── FEDformer-lite — Frequency Enhanced Decomposed Transformer (Zhou 2022) ──
# Compact, contract-matching ((B,T,C)→(B,n_outputs)) reimplementation of the two
# core FEDformer ideas, ported from the official repo
# (PatchTST/.../FEDformer/layers/{FourierCorrelation,Autoformer_EncDec}.py):
#   (1) MOEDecomp — mixture of moving-average experts for seasonal/trend split.
#   (2) Fourier Enhanced Block — FFT → keep selected modes → complex-linear → iFFT,
#       giving a global frequency-domain "view" that resists the long-horizon
#       distribution drift / mean-reversion of point-wise attention.
# Encoder-only (no seq2seq decoder) to fit the shared single-output contract,
# same compaction philosophy as the Autoformer-lite / Informer-lite above.
class _SeriesDecompMulti(nn.Module):
    """MOEDecomp: softmax-weighted blend of several moving-average trend kernels."""
    def __init__(self, d_model, kernels=(4, 8, 16, 24)):
        super().__init__()
        self.kernels = list(kernels)
        self.avgs = nn.ModuleList([nn.AvgPool1d(k, stride=1, padding=0) for k in self.kernels])
        self.gate = nn.Linear(d_model, len(self.kernels))
    def _ma(self, x, avg, k):                       # x: (B,T,C)
        front = x[:, :1, :].repeat(1, (k - 1) - (k - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (k - 1) // 2, 1)
        xp = torch.cat([front, x, end], dim=1)
        t = avg(xp.transpose(1, 2)).transpose(1, 2)
        return t[:, :x.size(1), :]
    def forward(self, x):                           # x: (B,T,C)
        trends = [self._ma(x, avg, k).unsqueeze(-1) for avg, k in zip(self.avgs, self.kernels)]
        trends = torch.cat(trends, dim=-1)          # (B,T,C,n_kernels)
        w = torch.softmax(self.gate(x), dim=-1).unsqueeze(2)   # (B,T,1,n_kernels)
        trend = (trends * w).sum(dim=-1)            # (B,T,C)
        return x - trend, trend                     # seasonal, trend


class _FourierEnhancedBlock(nn.Module):
    """FEB-f: learnable complex linear map over a fixed set of low-frequency modes."""
    def __init__(self, d_model, seq_len, modes=32):
        super().__init__()
        self.modes = max(1, min(modes, seq_len // 2))
        scale = 1.0 / (d_model * d_model)
        # complex kernel: (d_model, d_model, modes)
        self.weight = nn.Parameter(scale * torch.rand(d_model, d_model, self.modes, dtype=torch.cfloat))
    def forward(self, x):                           # x: (B,T,d_model)
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)               # (B, T//2+1, D)
        m = min(self.modes, xf.size(1))
        out_f = torch.zeros_like(xf)
        # einsum over channel: (B,m,Din),(Din,Dout,m)->(B,m,Dout)
        sel = xf[:, :m, :]
        w = self.weight[:, :, :m]
        out_f[:, :m, :] = torch.einsum("bmi,iom->bmo", sel, w)
        return torch.fft.irfft(out_f, n=T, dim=1)   # (B,T,d_model)


class FEDformerLite(nn.Module):
    def __init__(self, n_features, lookback, d_model, n_layers, d_ff,
                 n_outputs=1, dropout=0.2, modes=32, **kw):
        super().__init__()
        self.revin  = RevIN(n_features)
        self.proj   = nn.Linear(n_features, d_model)
        self.decomp = _SeriesDecompMulti(d_model)
        n_blocks = max(n_layers, 2)
        self.feb   = nn.ModuleList([_FourierEnhancedBlock(d_model, lookback, modes)
                                    for _ in range(n_blocks)])
        self.norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ffn   = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)) for _ in range(n_blocks)])
        self.head  = _make_head(d_model, d_ff, n_outputs, dropout)
    def forward(self, x):
        x = self.revin(x)                           # (B,T,C)
        h = self.proj(x)                            # (B,T,d_model)
        seasonal, trend = self.decomp(h)
        s = seasonal
        for feb, norm, ffn in zip(self.feb, self.norm, self.ffn):
            s = norm(s + feb(s))                     # frequency-domain mixing + residual
            s = norm(s + ffn(s))                     # channel FFN + residual
        h = s + trend                                # recombine seasonal + trend
        return self.head(h.mean(dim=1))


# ── Time-LLM — reprogram a FROZEN GPT-2 for forecasting (Jin 2024) ──
# Patches of the series are "reprogrammed" into the LLM's embedding space via
# cross-attention to text-prototype embeddings, prepended with a Prompt-as-Prefix
# token built from the lookback window's stats (past-only → leakage-safe), then
# passed through frozen GPT-2 blocks. Only the reprogramming layer + prefix + head
# train; the GPT-2 backbone stays frozen. Heaviest model → run with few seeds.
class TimeLLM(nn.Module):
    def __init__(self, n_features, lookback, patch_len=16, n_outputs=1, dropout=0.2,
                 llm_name="gpt2", n_proto=256, n_heads=8, n_llm_layers=6, **kw):
        super().__init__()
        from transformers import GPT2Model            # lazy: only needed for this model
        self.revin     = RevIN(n_features)
        self.patch_len = patch_len
        self.n_features = n_features
        gpt2 = GPT2Model.from_pretrained(llm_name)
        gpt2.h = nn.ModuleList(list(gpt2.h[:n_llm_layers]))   # truncate for compute
        for p in gpt2.parameters(): p.requires_grad = False   # FREEZE backbone
        self.gpt2  = gpt2
        self.d_llm = gpt2.config.n_embd                       # 768
        vocab      = gpt2.config.vocab_size
        # Text prototypes: trainable, initialised from REAL word embeddings
        idx = torch.randint(0, vocab, (n_proto,))
        self.prototypes = nn.Parameter(gpt2.wte.weight[idx].detach().clone())  # (n_proto,d_llm)
        # Patch embedding (all features flattened per patch, like PatchTST)
        self.patch_proj = nn.Linear(patch_len * n_features, self.d_llm)
        # Reprogramming cross-attention (patch queries → prototype keys/values)
        self.nh = n_heads; self.dk = self.d_llm // n_heads
        self.q  = nn.Linear(self.d_llm, self.d_llm)
        self.k  = nn.Linear(self.d_llm, self.d_llm)
        self.v  = nn.Linear(self.d_llm, self.d_llm)
        self.drop = nn.Dropout(dropout)
        # Prompt-as-Prefix: 6 window stats → a prefix token in LLM space
        self.prefix = nn.Sequential(nn.Linear(6, self.d_llm), nn.GELU(),
                                    nn.Linear(self.d_llm, self.d_llm))
        self.head = _make_head(self.d_llm, self.d_llm, n_outputs, dropout)
    def _reprogram(self, patches):                    # patches: (B,P,d_llm)
        B, P, _ = patches.shape
        proto = self.prototypes.unsqueeze(0).expand(B, -1, -1)         # (B,n_proto,d_llm)
        q = self.q(patches).view(B, P, self.nh, self.dk).transpose(1, 2)
        k = self.k(proto).view(B, -1, self.nh, self.dk).transpose(1, 2)
        v = self.v(proto).view(B, -1, self.nh, self.dk).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk), dim=-1)
        out  = torch.matmul(self.drop(attn), v).transpose(1, 2).reshape(B, P, self.d_llm)
        return out
    def forward(self, x):
        x = self.revin(x)                                   # (B,T,C)
        B, T, C = x.shape
        npat = T // self.patch_len
        xp = x[:, :npat*self.patch_len, :].reshape(B, npat, self.patch_len*C)
        patches = self.patch_proj(xp)                       # (B,P,d_llm)
        patches = self._reprogram(patches)                  # reprogram into LLM space
        # Prompt-as-Prefix stats (per-batch, from the past window only)
        xm = x.mean(dim=2)                                  # (B,T) feature-averaged
        t  = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).expand(B, -1)
        slope = ((xm - xm.mean(1, keepdim=True)) * (t - 0.5)).mean(1)
        stats = torch.stack([xm.mean(1), xm.std(1), xm.amin(1), xm.amax(1),
                             slope, xm[:, -1]], dim=1)       # (B,6)
        pref = self.prefix(stats).unsqueeze(1)              # (B,1,d_llm)
        seq  = torch.cat([pref, patches], dim=1)            # (B,1+P,d_llm)
        out  = self.gpt2(inputs_embeds=seq).last_hidden_state
        return self.head(out.mean(dim=1))

# ── Loss Functions ────────────────────────────────────────────

class MSESqueeze(nn.Module):
    """MSE that squeezes the (B,1) head output to (B,) — avoids the
    (B,1) vs (B,) broadcasting bug in nn.MSELoss."""
    def forward(self, pred, target):
        return nn.functional.mse_loss(pred.squeeze(-1), target.squeeze())

class HuberLoss(nn.Module):
    def __init__(self, delta=1.5): super().__init__(); self.delta = delta
    def forward(self, pred, target):
        err = (pred.squeeze(-1) - target.squeeze()).abs()
        return torch.where(err < self.delta,
                           0.5*err**2, self.delta*(err-0.5*self.delta)).mean()

class QuantileLoss(nn.Module):
    def __init__(self, quantiles): super().__init__(); self.q = torch.tensor(quantiles, dtype=torch.float32)
    def forward(self, pred, target):
        self.q = self.q.to(pred.device)
        target = target.unsqueeze(1).expand_as(pred)
        err    = target - pred
        return torch.max((self.q-1)*err, self.q*err).mean()

class StudentTNLL(nn.Module):
    """Negative log-likelihood of a Student-t predictive distribution.
    Heavy-tailed → robust to the noisy, fat-tailed long-horizon returns.
    pred: (B, 3) = [mu, raw_sigma, raw_nu].  Point forecast = mu.
      sigma = softplus(raw_sigma) + eps          (scale > 0)
      nu    = softplus(raw_nu)    + 2.0          (dof > 2 → finite variance)"""
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, pred, target):
        mu     = pred[:, 0]
        sigma  = nn.functional.softplus(pred[:, 1]) + self.eps
        nu     = nn.functional.softplus(pred[:, 2]) + 2.0
        target = target.squeeze()
        z      = (target - mu) / sigma
        log_pdf = (torch.lgamma((nu+1)/2) - torch.lgamma(nu/2)
                   - 0.5*torch.log(nu*math.pi) - torch.log(sigma)
                   - 0.5*(nu+1)*torch.log1p(z*z/nu))
        return -log_pdf.mean()

def get_loss_fn(horizon_key, horizons_dict):
    loss_type = horizons_dict[horizon_key]["loss"]
    if loss_type == "mse":      return MSESqueeze()
    if loss_type == "huber":    return HuberLoss(delta=1.5)
    if loss_type == "quantile": return QuantileLoss(QUANTILES)
    if loss_type == "studentt": return StudentTNLL()
    raise ValueError(f"Unknown loss: {loss_type}")

def n_outputs_for(cfg):
    # Phase 3 (Ablation D / seq-path): explicit override emits a length-h path.
    if cfg.get("n_outputs_override"):
        return int(cfg["n_outputs_override"])
    if cfg["loss"] == "quantile": return cfg["n_quantiles"]
    if cfg["loss"] == "studentt": return 3            # mu, raw_sigma, raw_nu
    return 1

# Models that accept a `patch_len` arg (PatchTST family); others ignore it.
MODEL_REGISTRY = {
    "PatchTST":      PatchTST,
    "TCN":           TCN,
    "TCNBiLSTM":     TCNBiLSTM,
    "TimeMixer":     TimeMixer,
    "NHiTS":         NHiTS,
    "Informer":      Informer,
    "Autoformer":    Autoformer,
    "iTransformer":  iTransformer,
    "TFT":           TFT,           # C1: Temporal Fusion Transformer
    "CNNBiLSTMAttn": CNNBiLSTMAttn, # C2: CNN-BiLSTM with attention
    "FEDformer":     FEDformerLite, # C3: Frequency Enhanced Decomposed (compact)
    "TimeLLM":       TimeLLM,
}

def build_model(horizon_key, n_features, horizons_dict, model_name="PatchTST"):
    """Build any registered architecture. All share the (B,T,C)→(B,n_outputs)
    contract, RevIN, and the shared regression head, so the rest of the
    pipeline (losses, shrinkage, conformal CI) is model-agnostic."""
    cfg     = horizons_dict[horizon_key]
    dropout = cfg.get("dropout", 0.2)
    n_out   = n_outputs_for(cfg)
    d_model = cfg["d_model"]
    n_heads = 16 if d_model % 16 == 0 else (8 if d_model % 8 == 0 else 4)
    Cls = MODEL_REGISTRY.get(model_name)
    if Cls is None:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Choices: {list(MODEL_REGISTRY)}")
    common = dict(n_features=n_features, lookback=cfg["lookback"], d_model=d_model,
                  n_layers=cfg["layers"], d_ff=d_model*4, n_outputs=n_out, dropout=dropout)
    if model_name == "PatchTST":
        model = PatchTST(patch_len=cfg["patch"], n_heads=n_heads, **common)
    elif model_name == "TimeLLM":
        model = Cls(patch_len=cfg["patch"], n_heads=8, **common)   # frozen GPT-2 inside
    elif model_name in ("Informer", "Autoformer", "iTransformer"):
        model = Cls(n_heads=n_heads, **common)
    elif model_name in ("TFT", "CNNBiLSTMAttn"):
        # TFT + CNNBiLSTMAttn accept optional n_heads; use 4 or d_model-compatible default
        n_h4 = 4 if d_model % 4 == 0 else 1
        model = Cls(n_heads=n_h4, **common)
    else:                                              # TCN / TCNBiLSTM / TimeMixer / NHiTS / FEDformer
        model = Cls(**common)
    return model.to(DEVICE)

def get_configs(horizon_key, horizons_dict):
    # Single paper-aligned config; ensemble diversity comes from seeds only.
    base = horizons_dict[horizon_key]
    return [{"d_model": base["d_model"], "n_layers": base["layers"], "patch": base["patch"]}]

# ── Dataset ───────────────────────────────────────────────────

class WTIDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, lookback, sample_weights=None, scaler=None):
        # Leakage discipline: fit the RobustScaler on the TRAIN split only, then
        # pass that fitted scaler into the val/test/in-sample datasets. If no
        # scaler is supplied we fit one here (back-compat for one-off use) and
        # expose it as self.scaler so the caller can reuse it on later splits.
        self.lookback = lookback
        feats   = df[feature_cols].values.astype(np.float32)
        targets = df[target_col].values.astype(np.float32)
        if scaler is None:
            scaler = RobustScaler()
            feats  = np.clip(scaler.fit_transform(feats), -10, 10)
        else:
            feats  = np.clip(scaler.transform(feats), -10, 10)
        self.scaler = scaler
        self.X, self.y, self.weights = [], [], []
        for i in range(lookback, len(df)-1):
            y_val = targets[i]
            if not np.isnan(y_val) and np.isfinite(y_val):
                self.X.append(feats[i-lookback:i])
                self.y.append(y_val)
                self.weights.append(float(sample_weights[i]) if sample_weights is not None else 1.0)
        self.X       = np.array(self.X, dtype=np.float32)
        self.y       = np.array(self.y, dtype=np.float32)
        self.weights = np.array(self.weights, dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

def build_sample_weights(df, crises, recency_factor=2.0, crisis_factor=3.0, neg_date="2020-04-20"):
    n = len(df)
    w = np.linspace(1.0, recency_factor, n).astype(np.float32)
    for cs, ce in crises.values():
        mask = (df.index >= cs) & (df.index <= ce)
        mask &= (df.index != pd.Timestamp(neg_date))
        w[mask] *= crisis_factor
    return w

# ── Training (with AMP mixed precision) ───────────────────────

_FFT_FP32_MODELS = {"FreEformer", "TiDE"}  # cuFFT requires power-of-2 sizes in fp16

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, grad_clip=1.0):
    _amp_ok = torch.cuda.is_available() and type(model).__name__ not in _FFT_FP32_MODELS
    model.train(); total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE, non_blocking=IS_CUDA), y.to(DEVICE, non_blocking=IS_CUDA)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=_amp_ok):
            loss = loss_fn(model(X), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        if IS_TPU:                           # XLA: flush computation graph each step
            try:
                import torch_xla.core.xla_model as xm; xm.mark_step()
            except ImportError:
                pass
        total += loss.item()*len(y)
    return total / len(loader.dataset)

def evaluate(model, loader, loss_fn):
    _amp_ok = torch.cuda.is_available() and type(model).__name__ not in _FFT_FP32_MODELS
    model.eval(); total = 0.0; preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE, non_blocking=IS_CUDA), y.to(DEVICE, non_blocking=IS_CUDA)
            with torch.cuda.amp.autocast(enabled=_amp_ok):
                out = model(X)
            total += loss_fn(out, y).item()*len(y)
            preds.append(out.float().cpu()); targets.append(y.cpu())
    return total/len(loader.dataset), torch.cat(preds), torch.cat(targets)

def train_model(model, train_ds, val_ds, horizon_key, horizons_dict, seed=42, verbose=True):
    set_seed(seed)
    loss_fn    = get_loss_fn(horizon_key, horizons_dict)
    w          = torch.from_numpy(train_ds.weights)
    sampler    = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    # num_workers stays 0: WTIDataset pre-materialises all windows in RAM, so
    # __getitem__ is a trivial index — workers would add only IPC/spawn overhead.
    # pin_memory overlaps the host→device copy with compute (see non_blocking below).
    tr_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=IS_CUDA)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   pin_memory=IS_CUDA)
    wd         = horizons_dict[horizon_key].get("weight_decay", 1e-4)   # tunable per horizon
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR*0.1)
    # Disable GradScaler for models with complex-dtype parameters (e.g. FEDformer/FEDformerW
    # use cfloat weights for Fourier mixing). GradScaler.unscale_() does not support
    # ComplexFloat gradients and crashes; keeping autocast is still fine.
    _has_complex = any(p.is_complex() for p in model.parameters())
    _no_amp      = type(model).__name__ in _FFT_FP32_MODELS
    scaler     = torch.cuda.amp.GradScaler(enabled=IS_CUDA and not _has_complex and not _no_amp)
    best_val, best_state, patience_cnt, history = float("inf"), None, 0, []
    for epoch in range(1, MAX_EPOCHS+1):
        tl = train_one_epoch(model, tr_loader, optimizer, loss_fn, scaler)
        vl, _, _ = evaluate(model, val_loader, loss_fn)
        scheduler.step(); history.append(vl)
        if vl < best_val:
            best_val = vl; best_state = deepcopy(model.state_dict()); patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose: print(f"    Early stop epoch {epoch} (best val={best_val:.6f})")
                break
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  train={tl:.6f}  val={vl:.6f}")
    model.load_state_dict(best_state)
    return model, history

# ============================================================
# MULTI-HORIZON JOINT MODEL
# A single PatchTST backbone with three separate prediction heads
# (7d / 90d / 180d). Training a shared encoder forces it to learn
# representations useful across all timescales, which reduces the
# degradation gap between short-horizon and long-horizon accuracy.
# Used by _forecast_template_mh.py + 03_forecast_multi.py.
# ============================================================

class MultiHorizonPatchTST(nn.Module):
    """Single PatchTST encoder backbone → three per-horizon prediction heads.

    Input:  (B, T, C)
    Output: dict {"7d": (B, n_out_7d), "90d": (B, n_out_90d), "180d": (B, n_out_180d)}

    n_out_7d  = 1  (MSE / point)
    n_out_90d = 3  (StudentT: mu, raw_sigma, raw_nu)
    n_out_180d= 3  (StudentT: same structure)
    """
    def __init__(self, n_features, lookback, patch_len, d_model, n_heads, n_layers, d_ff,
                 n_out_7d=1, n_out_90d=3, n_out_180d=3, dropout=0.2, **kw):
        super().__init__()
        self.revin     = RevIN(n_features)
        self.patch_emb = PatchEmbedding(patch_len, n_features, d_model, dropout)
        self.pos_enc   = PositionalEncoding(d_model, max_len=lookback // patch_len + 8,
                                            dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Three separate heads — shared backbone, independent outputs
        self.head_7d   = _make_head(d_model, d_ff, n_out_7d,   dropout)
        self.head_90d  = _make_head(d_model, d_ff, n_out_90d,  dropout)
        self.head_180d = _make_head(d_model, d_ff, n_out_180d, dropout)

    def forward(self, x):
        x      = self.revin(x)
        x      = self.patch_emb(x)
        x      = self.pos_enc(x)
        pooled = self.encoder(x).mean(dim=1)   # (B, d_model)
        return {
            "7d":   self.head_7d(pooled),
            "90d":  self.head_90d(pooled),
            "180d": self.head_180d(pooled),
        }


def build_multi_model(n_features, horizons_dict):
    """Build a MultiHorizonPatchTST using the 180d backbone config (longest
    lookback + most capacity). Register it in MODEL_REGISTRY under
    'MultiPatchTST' so template discovery works."""
    cfg     = horizons_dict["180d"]
    d_model = cfg["d_model"]
    n_heads = 16 if d_model % 16 == 0 else (8 if d_model % 8 == 0 else 4)
    model   = MultiHorizonPatchTST(
        n_features  = n_features,
        lookback    = cfg["lookback"],
        patch_len   = cfg["patch"],
        d_model     = d_model,
        n_heads     = n_heads,
        n_layers    = cfg["layers"],
        d_ff        = d_model * 4,
        n_out_7d    = 1,    # MSE head
        n_out_90d   = 3,    # StudentT head
        n_out_180d  = 3,    # StudentT head
        dropout     = cfg.get("dropout", 0.25),
    )
    return model.to(DEVICE)


# ── Multi-horizon Dataset ────────────────────────────────────────────────────
class WTIDatasetMH(Dataset):
    """Dataset that loads three target columns simultaneously.

    Returns (X, y_dict) where y_dict is {"7d": scalar, "90d": scalar, "180d": scalar}.
    NaN targets are allowed; the training loop applies per-sample masking so the
    backbone is still trained on all samples that have at least a valid 7d target.
    """
    def __init__(self, df, feature_cols, target_cols, lookback, sample_weights=None, scaler=None):
        # target_cols: dict {"7d": "target_7d", "90d": "target_90d", "180d": "target_180d"}
        self.lookback = lookback
        feats = df[feature_cols].values.astype(np.float32)
        if scaler is None:
            scaler = RobustScaler()
            feats  = np.clip(scaler.fit_transform(feats), -10, 10)
        else:
            feats  = np.clip(scaler.transform(feats), -10, 10)
        self.scaler = scaler

        tgt = {hk: df[col].values.astype(np.float32)
               for hk, col in target_cols.items()}

        self.X, self.y, self.weights = [], {hk: [] for hk in tgt}, []
        for i in range(lookback, len(df) - 1):
            # Include sample if the 7d target is valid (primary horizon)
            if not np.isnan(tgt["7d"][i]) and np.isfinite(tgt["7d"][i]):
                self.X.append(feats[i - lookback:i])
                for hk in tgt:
                    self.y[hk].append(tgt[hk][i])
                self.weights.append(
                    float(sample_weights[i]) if sample_weights is not None else 1.0)

        self.X       = np.array(self.X, dtype=np.float32)
        self.y       = {hk: np.array(v, dtype=np.float32) for hk, v in self.y.items()}
        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = {hk: torch.tensor(self.y[hk][idx]) for hk in self.y}
        return x, y


import pandas as pd  # needed by build_sample_weights neg_date comparison
print("✅ 02_model_shared loaded (RevIN + AMP, paper-aligned)")
