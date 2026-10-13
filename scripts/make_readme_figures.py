"""The README figures: a held-out daily risk map and the skill comparison.

    python scripts/make_readme_figures.py [YYYY-MM-DD] [--dev-root PATH]   # default 2022-07-13

The map replays one day of the test period through ``GridRiskService`` with the
model trained only on the years before 2022 (``firerisk_daily_current_20220105``),
so the day is out of sample. Replay needs the full weather store
(``outputs/store/portugal/`` built by stage 1d) and that harness model; neither is
in this repository, so point ``--dev-root`` at a project that has them (the shipped
PNGs were made that way). The skill chart needs nothing.

Writes ``assets/{risk_map,skill}_{light,dark}.png``.
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA_ROOT = ROOT                     # where the store + harness model live (--dev-root)

# Sequential blue ramp (light -> dark) and the chart surfaces / inks per mode.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RAMP_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
MODES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
              "ramp": RAMP, "fire": "#d9622b", "ours": "#2a78d6", "other": "#a3a29c"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#383835",
             "ramp": RAMP_DARK, "fire": "#f08a4b", "ours": "#3987e5", "other": "#6b6a64"},
}

# Daily PR-AUC on every cell-day of 2022-01-05..2023-12-31 (outputs/models/grid_evaluation_*).
SKILL = [("Random", 0.019), ("FWI alone", 0.029), ("Previous model (xgb18)", 0.083),
         ("FWI × cell fire history", 0.147), ("firerisk_daily", 0.234)]


def risk_map(day: str) -> None:
    sys.path.insert(0, str(DATA_ROOT))
    from app.api.grid import GridRiskService

    model = DATA_ROOT / "outputs" / "models" / "grid" / "harness" / "firerisk_daily_current_20220105.joblib"
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svc = GridRiskService(str(model), feed="store", state_dir=Path(d))
        run = svc.run(day, save=False)
    t = pd.Timestamp(day)
    today = run.daily[run.daily["day"] == t].set_index("cell")
    ev = svc.ctx.events
    burned = np.unique(ev.loc[(pd.to_datetime(ev["day"]) == t) & (ev["cell"] >= 0), "cell"])
    top = today.sort_values("probability", ascending=False)
    in_top = int(np.isin(burned, top.index[:100]).sum())
    for mode, c in MODES.items():
        fig, ax = plt.subplots(figsize=(6.4, 9.0), dpi=150)
        fig.patch.set_facecolor(c["surface"])
        ax.set_facecolor(c["surface"])
        cmap = LinearSegmentedColormap.from_list("seq", c["ramp"])
        norm = LogNorm(vmin=0.002, vmax=0.4)
        p = today["probability"].clip(0.002, 0.4)
        ax.scatter(today["slon"], today["slat"], c=p, cmap=cmap, norm=norm, marker="s", s=24, linewidths=0)
        f = today.loc[burned]
        ax.scatter(f["slon"], f["slat"], s=30, facecolors="none", edgecolors=c["fire"], linewidths=1.2,
                   label=f"cells with a fire that day ({len(burned)})")
        ax.set_aspect(1 / np.cos(np.radians(39.6)))
        ax.axis("off")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02,
                          ticks=[0.002, 0.01, 0.05, 0.2])
        cb.ax.set_yticklabels(["0.2 %", "1 %", "5 %", "20 %"], color=c["ink2"], fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(color=c["grid"])
        cb.set_label("P(at least one fire in the cell that day)", color=c["ink2"], fontsize=9)
        leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
        for txt in leg.get_texts():
            txt.set_color(c["ink2"])
        ax.set_title(f"Daily fire risk, {t:%d %B %Y}", color=c["ink"], fontsize=13, loc="left", pad=12)
        ax.text(0.0, 1.0, f"held-out day · {in_top} of {len(burned)} burning cells were in the model's top 100 of 1,036",
                transform=ax.transAxes, color=c["ink2"], fontsize=8.5, va="bottom")
        fig.savefig(ASSETS / f"risk_map_{mode}.png", facecolor=c["surface"], bbox_inches="tight")
        plt.close(fig)
    print(f"risk map {day}: {len(burned)} fire cells, {in_top} in the top 100")


def skill() -> None:
    names, vals = zip(*SKILL)
    for mode, c in MODES.items():
        fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=150)
        fig.patch.set_facecolor(c["surface"])
        ax.set_facecolor(c["surface"])
        colors = [c["ours"] if n == "firerisk_daily" else c["other"] for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, height=0.62, color=colors, edgecolor=c["surface"], linewidth=2)
        for yi, v, n in zip(y, vals, names):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", color=c["ink"] if n == "firerisk_daily" else c["ink2"],
                    fontsize=10, fontweight="bold" if n == "firerisk_daily" else "normal")
        ax.set_yticks(y, names, color=c["ink"], fontsize=10)
        ax.set_xlim(0, 0.27)
        ax.tick_params(axis="x", colors=c["ink2"], labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("daily PR-AUC, every cell-day of 2022–23 (base rate 1.9 %)", color=c["ink2"], fontsize=9)
        fig.tight_layout()
        fig.savefig(ASSETS / f"skill_{mode}.png", facecolor=c["surface"])
        plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--dev-root" in args:
        i = args.index("--dev-root")
        DATA_ROOT = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    ASSETS.mkdir(exist_ok=True)
    skill()
    risk_map(args[0] if args else "2022-07-13")
