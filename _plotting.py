# ════════════════════════════════════════════════════════════════════
#  FILE : _plotting.py
#  ROLE : Business-grade Plotly theme + figure/table/animation savers
# ════════════════════════════════════════════════════════════════════
#  WHAT — One centralized place for ALL plotting. Registers a clean
#         business template and exposes save_fig / save_table /
#         save_animation / add_crises so every stage draws consistent,
#         presentation-quality, non-overlapping figures.
#
#  WHY  — The old code used matplotlib in every file with fixed sizes,
#         manual legend positions (overlap-prone) and inconsistent
#         palettes. The thesis needs crisp, business-deck-level visuals.
#         Plotly gives interactive HTML (for the defense) + high-DPI PNG
#         (for the document) from a single figure, with auto-spacing
#         that prevents legend/label overlap.
#
#  HOW  — apply_theme() sets a global pio template (font, palette,
#         generous margins, legend in a gutter outside the plot area).
#         save_fig() writes <name>.html (interactive) AND <name>.png
#         (kaleido, scale=3). save_animation() builds a play/slider
#         figure, writes the HTML, a representative still PNG, and an
#         optional GIF (imageio). Every export is guarded so a missing
#         kaleido/imageio only disables that format — never crashes a run.
#
#  READS    : globals FIGURES_DIR/TABLES_DIR (from _env.py), CRISES
#  WRITES   : figures/<name>.html|.png , tables/<name>.csv + go.Table
#  COMPUTE  : CPU-ONLY
#  PIPELINE : exec'd right after _env.py; used by every plotting stage
#  RUNTIME  : ~0.1–2 s per figure (PNG export dominates)
# ════════════════════════════════════════════════════════════════════

from pathlib import Path as _Path
import pandas as _pd

# ── Optional deps (guarded so absence only disables a format) ────────
try:
    import plotly.graph_objects as go
    import plotly.io as _pio
    from plotly.subplots import make_subplots          # noqa: F401  (re-exported)
    _PLOTLY_OK = True
except Exception:                                       # pragma: no cover
    import subprocess as _sp, sys as _sys
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", "plotly", "kaleido"], check=False)
    try:
        import plotly.graph_objects as go
        import plotly.io as _pio
        from plotly.subplots import make_subplots       # noqa: F401
        _PLOTLY_OK = True
    except Exception:
        _PLOTLY_OK = False
        print("⚠️  plotly unavailable — _plotting.py is a no-op shim")

try:
    import kaleido  # noqa: F401
    _KALEIDO_OK = True
except Exception:
    # kaleido is needed for PNG/PDF export — auto-install once (same pattern as plotly above).
    # On Kaggle it isn't pre-installed; on local machines it often gets missed.
    try:
        import subprocess as _sp2, sys as _sys2
        _sp2.run([_sys2.executable, "-m", "pip", "install", "-q", "kaleido"], check=False)
        import kaleido  # noqa: F401
        _KALEIDO_OK = True
        print("  ✅ kaleido installed and loaded (PNG/PDF export enabled)")
    except Exception:
        _KALEIDO_OK = False
        print("  ⚠️  kaleido unavailable — PNG/PDF export disabled (HTML figures still work)")

try:
    import imageio.v2 as _imageio
    _IMAGEIO_OK = True
except Exception:
    # GIF/MP4 export is wanted — auto-install once. imageio-ffmpeg bundles an
    # ffmpeg binary so MP4 works without a separate system install. Same self-
    # bootstrap pattern 00/10 use; stays quiet and leaves _IMAGEIO_OK=False if it
    # can't install (HTML animations + PNGs are unaffected either way).
    try:
        import subprocess as _sp_io, sys as _sys_io
        print("  [plotting] installing imageio + imageio-ffmpeg for GIF/MP4 export …")
        _sp_io.run([_sys_io.executable, "-m", "pip", "install", "-q",
                    "imageio", "imageio-ffmpeg"], check=False)
        import imageio.v2 as _imageio
        _IMAGEIO_OK = True
    except Exception as _io_e:
        _IMAGEIO_OK = False
        print(f"  [plotting] imageio unavailable ({_io_e}); GIF/MP4 off, HTML+PNG unaffected.")

# ── Output folders (from _env.py if present, else sensible default) ──
_FIG_DIR = globals().get("FIGURES_DIR", _Path("outputs/all/figures"))
_TAB_DIR = globals().get("TABLES_DIR",  _Path("outputs/all/tables"))
for _d in (_FIG_DIR, _TAB_DIR):
    _Path(_d).mkdir(parents=True, exist_ok=True)

# ── Business palette + crisis windows ────────────────────────────────
# Reuse the project palette if 00_setup already defined one, then merge in any
# keys it's missing so apply_theme() always has "grid", "actual", etc.
_PALETTE_DEFAULTS = {
    "actual": "#1f3b73", "pred": "#d1495b", "interval": "#b9c6e0",
    "accent": "#2a9d8f", "muted": "#8d99ae", "grid": "#e6e9ef",
}
PLOT_PALETTE = {**_PALETTE_DEFAULTS, **globals().get("PALETTE", {})}
_SERIES_COLORS = ["#1f3b73", "#d1495b", "#2a9d8f", "#e9a23b", "#6a4c93",
                  "#118ab2", "#ef476f", "#06d6a0", "#8d99ae", "#073b4c"]
# Standard crisis windows (reuse 00_setup's CRISES if available)
_CRISES = globals().get("CRISES", {
    "GFC": ("2008-09-01", "2009-06-30"),
    "OPEC War": ("2014-06-01", "2016-03-31"),
    "COVID": ("2020-02-01", "2020-09-30"),
})
_CRISIS_FILL = {"GFC": "#d62728", "OPEC War": "#ff7f0e", "COVID": "#9467bd"}

# Thesis-document figure size (16:9, generous for two-column print)
FIG_W, FIG_H, FIG_SCALE = 1280, 720, 3


def apply_theme():
    """Register and activate the DDP business template (idempotent)."""
    if not _PLOTLY_OK:
        return
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
                  size=15, color="#1a1a1a"),
        title=dict(font=dict(size=22, color="#0b1f3a"), x=0.02, xanchor="left"),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=_SERIES_COLORS,
        margin=dict(l=70, r=180, t=70, b=70),          # right gutter for legend
        # legend OUTSIDE the plot area → never overlaps the data
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="#d0d4dc",
                    borderwidth=1, x=1.02, xanchor="left", y=1, yanchor="top"),
        xaxis=dict(showgrid=True, gridcolor=PLOT_PALETTE["grid"], zeroline=False,
                   linecolor="#9aa3b2", ticks="outside"),
        yaxis=dict(showgrid=True, gridcolor=PLOT_PALETTE["grid"], zeroline=False,
                   linecolor="#9aa3b2", ticks="outside"),
        hovermode="x unified",
    )
    _pio.templates["ddp"] = tmpl
    _pio.templates.default = "plotly_white+ddp"


def add_crises(fig, row=None, col=None):
    """Shade the GFC / OPEC-War / COVID windows on a time-axis figure
    (replaces the matplotlib shade_crises helper)."""
    if not _PLOTLY_OK:
        return fig
    kw = {}
    if row is not None:
        kw = dict(row=row, col=col or 1)
    for name, (s, e) in _CRISES.items():
        fig.add_vrect(x0=s, x1=e, fillcolor=_CRISIS_FILL.get(name, "#999"),
                      opacity=0.10, line_width=0,
                      annotation_text=name, annotation_position="top left",
                      annotation_font_size=10, **kw)
    return fig


def save_fig(fig, name, html=True, png=True, pdf=False):
    """Write a Plotly figure to figures/<name>.{html,png,pdf}.
    HTML is always safe; PNG/PDF need kaleido (guarded)."""
    if not _PLOTLY_OK:
        print(f"  [plot] skipped '{name}' (plotly unavailable)")
        return
    apply_theme()
    base = _Path(_FIG_DIR) / name
    if html:
        try:
            fig.write_html(str(base) + ".html", include_plotlyjs="cdn",
                           full_html=True)
        except Exception as _e:
            print(f"  [warn] save_fig html '{name}': {_e}")
    if png and _KALEIDO_OK:
        try:
            fig.write_image(str(base) + ".png", width=FIG_W, height=FIG_H, scale=FIG_SCALE)
        except Exception as _e:
            print(f"  [warn] save_fig png '{name}': {_e}")
    elif png and not _KALEIDO_OK:
        print(f"  [plot] '{name}': PNG skipped (install kaleido); HTML written")
    if pdf and _KALEIDO_OK:
        try:
            fig.write_image(str(base) + ".pdf", width=FIG_W, height=FIG_H)
        except Exception as _e:
            print(f"  [warn] save_fig pdf '{name}': {_e}")
    print(f"  📊 saved figure: {base.name}")
    return str(base)


def save_table(df, name, title=None, fmt="{:.4f}"):
    """Write a DataFrame to tables/<name>.csv AND a styled go.Table
    figure (figures/<name>.{html,png}) for drop-in deck/thesis use."""
    df = _pd.DataFrame(df)
    csv_path = _Path(_TAB_DIR) / f"{name}.csv"
    try:
        df.to_csv(csv_path)
    except Exception as _e:
        print(f"  [warn] save_table csv '{name}': {_e}")
    if not _PLOTLY_OK:
        return str(csv_path)
    apply_theme()
    # format numeric cells, keep the index as the first column
    show = df.reset_index()
    cells = []
    for c in show.columns:
        col = show[c]
        if _pd.api.types.is_float_dtype(col):
            cells.append([fmt.format(v) if _pd.notna(v) else "" for v in col])
        else:
            cells.append([str(v) for v in col])
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in show.columns],
                    fill_color="#0b1f3a", font=dict(color="white", size=14),
                    align="left", height=32),
        cells=dict(values=cells, fill_color=[["#f5f7fb", "white"] * len(show)],
                   align="left", height=28, font=dict(size=13)))])
    fig.update_layout(title=title or name,
                      margin=dict(l=10, r=10, t=50, b=10))
    save_fig(fig, name, html=True, png=True)
    return str(csv_path)


def save_animation(fig, name, gif=False, fps=4):
    """Write an ANIMATED Plotly figure (one built with frames + a slider)
    to figures/<name>.html (interactive). Also writes a representative
    still PNG for the printed thesis, and — if gif=True and imageio +
    kaleido are present — a looping GIF for slides. All guarded.
    """
    if not _PLOTLY_OK:
        print(f"  [anim] skipped '{name}' (plotly unavailable)")
        return
    apply_theme()
    base = _Path(_FIG_DIR) / name
    # 1) interactive HTML (always)
    try:
        fig.write_html(str(base) + ".html", include_plotlyjs="cdn", full_html=True,
                       auto_play=False)
    except Exception as _e:
        print(f"  [warn] save_animation html '{name}': {_e}")
    # 2) still frame PNG (last frame = most informative end-state)
    if _KALEIDO_OK:
        try:
            frames = list(getattr(fig, "frames", []) or [])
            still = go.Figure(data=(frames[-1].data if frames else fig.data),
                              layout=fig.layout)
            still.layout.updatemenus = []; still.layout.sliders = []
            still.write_image(str(base) + ".png", width=FIG_W, height=FIG_H, scale=FIG_SCALE)
        except Exception as _e:
            print(f"  [warn] save_animation still '{name}': {_e}")
    # 3) optional GIF (per-frame render → imageio stitch)
    if gif and _KALEIDO_OK and _IMAGEIO_OK:
        try:
            frames = list(getattr(fig, "frames", []) or [])
            imgs = []
            for fr in frames:
                ff = go.Figure(data=fr.data, layout=fig.layout)
                ff.layout.updatemenus = []; ff.layout.sliders = []
                imgs.append(_imageio.imread(
                    ff.to_image(format="png", width=FIG_W, height=FIG_H, scale=1)))
            if imgs:
                _imageio.mimsave(str(base) + ".gif", imgs, fps=fps, loop=0)
                print(f"  🎞️  saved animation gif: {base.name}.gif")
        except Exception as _e:
            print(f"  [warn] save_animation gif '{name}': {_e}")
    elif gif:
        print(f"  [anim] '{name}': GIF skipped (need kaleido+imageio); HTML written")
    print(f"  🎬 saved animation: {base.name}.html")
    return str(base)


if _PLOTLY_OK:
    apply_theme()
print(f"✅ _plotting.py loaded  |  plotly={_PLOTLY_OK} kaleido={_KALEIDO_OK} "
      f"imageio={_IMAGEIO_OK}  |  figures→{_FIG_DIR}")
