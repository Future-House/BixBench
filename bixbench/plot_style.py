import urllib.request
from pathlib import Path
from tempfile import gettempdir

FONT_PATH = Path(gettempdir()) / "SometypeMono-Regular.ttf"


def _download_font():
    if not FONT_PATH.exists():
        urllib.request.urlretrieve(
            "https://github.com/googlefonts/sometype-mono/raw/refs/heads/master/fonts/ttf/SometypeMono-Regular.ttf",
            FONT_PATH,
        )


# Aim to have these colors be color-blind friendly, and avoid black/white
# We need at least 8 colors for a good color cycle
# SEE: https://davidmathlogic.com/colorblind/'s Tol theme or https://colorbrewer2.org/
COLOR_CYCLE = (
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion/Orange-red
    "#009E73",  # Teal green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#999999",  # Grey
    "#56147C",  # Deep Purple
    "#994F3D",  # Terracotta
)


def set_fh_mpl_style(dark_mode: bool = False):
    """Sets the FutureHouse plot style as a global matplotlib default."""
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Please `pip install matplotlib` to use set_fh_mpl_style."
        ) from None

    _download_font()

    fe = mpl.font_manager.FontEntry(fname=str(FONT_PATH), name="sometype")
    mpl.font_manager.fontManager.ttflist.append(fe)
    if dark_mode:
        mpl.rcParams.update({
            "axes.facecolor": "#000000",  # Black background for axes
            "grid.color": "#444444",  # Dark gray grid lines
            "axes.edgecolor": "#FFFFFF",  # White axes edges
            "figure.facecolor": "#000000",  # Black background for the figure
            "axes.grid": False,
            "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),
            "font.family": fe.name,
            "font.size": 14,
            "figure.figsize": (
                6,
                6 / 1.3,
            ),  # Adjust figure size for 1/3 the width of a presentation slide
            "figure.dpi": 200,
            "ytick.left": True,
            "xtick.bottom": True,
            "ytick.color": "#FFFFFF",  # White y-tick labels
            "xtick.color": "#FFFFFF",  # White x-tick labels
            "axes.labelcolor": "#FFFFFF",  # White axis labels
            "axes.titlecolor": "#FFFFFF",  # White title
            "text.color": "#FFFFFF",  # White text
            "image.cmap": "viridis",  # Colormap suitable for dark backgrounds
            "lines.markersize": 6,
        })
    else:
        mpl.rcParams.update({
            "axes.facecolor": "#FFF",
            "grid.color": "#AAAAAA",  # Dark gray grid lines
            "axes.edgecolor": "#333333",
            "figure.facecolor": "#FFFFFF",
            "axes.grid": False,
            "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),
            "font.family": fe.name,
            "font.size": 14,
            "figure.figsize": (
                6,
                6 / 1.3,
            ),  # Adjust figure size for 1/3 the width of a presentation slide
            "figure.dpi": 200,
            "ytick.left": True,
            "xtick.bottom": True,
            "image.cmap": "viridis",
            "lines.markersize": 6,
        })


def set_fh_plotly_style(dark_mode: bool = False):
    """Sets the FutureHouse plot style as a global plotly default.

    NOTE: I haven't figured out how to set Courier Prime as the font.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        raise ImportError(
            "Please `pip install plotly` to use set_fh_plotly_style."
        ) from None

    font_family = "Courier"

    if dark_mode:
        layout = {
            "font": {"family": font_family, "size": 14, "color": "#FFFFFF"},
            "plot_bgcolor": "#000000",
            "paper_bgcolor": "#000000",
            "xaxis": {
                "showgrid": True,
                "gridcolor": "#444444",
                "linecolor": "#FFFFFF",
                "tickfont": {"color": "#FFFFFF"},
                "titlefont": {"color": "#FFFFFF"},
                "color": "#FFFFFF",
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "#444444",
                "linecolor": "#FFFFFF",
                "tickfont": {"color": "#FFFFFF"},
                "titlefont": {"color": "#FFFFFF"},
                "color": "#FFFFFF",
            },
            "colorway": COLOR_CYCLE,
            "legend": {
                "font": {"family": font_family, "size": 14, "color": "#FFFFFF"},
                "bgcolor": "#000000",
                "bordercolor": "#FFFFFF",
                "borderwidth": 1,
                "itemsizing": "constant",
            },
            "title": {"font": {"family": font_family, "size": 16, "color": "#FFFFFF"}},
            "hoverlabel": {
                "font": {"family": font_family, "size": 14},
                "bgcolor": "#333333",
                "bordercolor": "#FFFFFF",
            },
        }
    else:
        layout = {
            "font": {"family": font_family, "size": 14, "color": "#000000"},
            "plot_bgcolor": "#FFFFFF",
            "paper_bgcolor": "#FFFFFF",
            "xaxis": {
                "showgrid": True,
                "gridcolor": "#AAAAAA",
                "linecolor": "#000000",
                "tickfont": {"color": "#000000"},
                "titlefont": {"color": "#000000"},
                "color": "#000000",
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "#AAAAAA",
                "linecolor": "#000000",
                "tickfont": {"color": "#000000"},
                "titlefont": {"color": "#000000"},
                "color": "#000000",
            },
            "colorway": COLOR_CYCLE,
            "legend": {
                "font": {"family": font_family, "size": 14, "color": "#000000"},
                "bgcolor": "#FFFFFF",
                "bordercolor": "#000000",
                "borderwidth": 1,
                "itemsizing": "constant",
            },
            "title": {"font": {"family": font_family, "size": 16, "color": "#000000"}},
            "hoverlabel": {
                "font": {"family": font_family, "size": 14},
                "bgcolor": "#FFFFFF",
                "bordercolor": "#000000",
            },
        }

    template = go.layout.Template()
    for k, v in layout.items():
        setattr(template.layout, k, v)
    pio.templates["futurehouse"] = template
    pio.templates.default = "futurehouse"
