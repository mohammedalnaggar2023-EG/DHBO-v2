"""
Build every table and figure for the paper from results/runs/*.json|npz.

    python make_tables.py [results_dir]

Writes results/tables/*.csv, results/figures/*.png and results/paper_tables.md
(all tables in Markdown, ready to paste into the manuscript).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRY_ORDER = ["India", "Brazil", "Russia", "USA", "France", "Germany", "Japan", "South Korea"]
MODEL_ORDER = ["Naive", "ARIMA", "SARIMA", "ETS (damped)", "LSTM", "GRU", "ARIMA-LSTM hybrid", "DHBO (proposed)"]
METRICS = ["MAE", "RMSE", "MAPE", "sMAPE", "MdAPE", "MASE"]

# reference palette (light mode): identity colours, fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_DHBO, C_JAIN, C_OTHER = "#2a78d6", "#eb6834", "#a8a7a2"


def load_runs(out: Path):
    runs = {}
    for f in sorted((out / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        runs[(r["target"], r["country"])] = r
    return runs


def fmt(v, metric):
    if metric in ("MAPE", "sMAPE", "MdAPE"):
        return f"{v:.2f}"
    if metric == "MASE":
        return f"{v:.3f}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"


def md_table(df: pd.DataFrame, bold_min_cols=()):
    """DataFrame -> Markdown with the column minimum in bold."""
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    mins = {c: df[c].min() for c in bold_min_cols if c in df}
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            s = v if isinstance(v, str) else fmt(v, str(c).split(" ")[-1] if " " in str(c) else str(c))
            if c in mins and not isinstance(v, str) and np.isclose(v, mins[c]):
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main(out="results"):
    out = Path(out)
    runs = load_runs(out)
    if not runs:
        print("no runs found")
        return
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    targets = sorted({t for t, _ in runs}, key=lambda t: ["daily", "cumulative"].index(t))
    md = ["# Tables generated from the experiment outputs\n"]

    # Table 1: data summary
    rows = []
    for (t, c), r in runs.items():
        d = r["dates"]
        rows.append({"Target": t, "Country": c, "Days": r["n"], "Start": d["start"], "Train end": d["train_end"],
                     "Test start": d.get("test_start", d["val_cal_end"]), "End": d["end"], "Test origins": r["n_origins"]["test"],
                     "SARIMA order": r["sarima_order"], "ARIMA order": r["arima_order"]})
    t1 = pd.DataFrame(rows).set_index("Country")
    t1.to_csv(tdir / "table1_data.csv")
    md += ["## Table 1. Data, splits and selected statistical orders\n", t1.to_markdown(), "\n"]

    for t in targets:
        cs = [c for c in COUNTRY_ORDER if (t, c) in runs]
        # Table 2: full metrics per country
        md.append(f"## Table 2-{t}. Test-set accuracy, {t} series (mean over origins and 1..H horizons)\n")
        for c in cs:
            m = pd.DataFrame(runs[(t, c)]["metrics"]).T.loc[MODEL_ORDER, METRICS]
            m.index.name = c
            m.to_csv(tdir / f"table2_{t}_{c}.csv")
            md += [md_table(m, METRICS), "\n"]
        # Table 3: MAPE + MASE summary across countries
        for metric in ("MAPE", "MASE"):
            s = pd.DataFrame({c: {k: runs[(t, c)]["metrics"][k][metric] for k in MODEL_ORDER} for c in cs})
            s["Mean"] = s.mean(axis=1)
            s.index.name = "Model"
            s.to_csv(tdir / f"table3_{t}_{metric}.csv")
            md += [f"## Table 3-{t}. {metric} by country, {t} series\n", md_table(s, list(s.columns)), "\n"]
        # Table 4: improvement over the ARIMA-LSTM hybrid
        imp = pd.DataFrame({c: {m_: 100 * (1 - runs[(t, c)]["metrics"]["DHBO (proposed)"][m_] /
                                         runs[(t, c)]["metrics"]["ARIMA-LSTM hybrid"][m_]) for m_ in METRICS}
                            for c in cs}).T
        imp.index.name = "Country"
        imp.to_csv(tdir / f"table4_{t}_improvement_vs_arima_lstm.csv")
        md += [f"## Table 4-{t}. % reduction in error of DHBO vs the ARIMA-LSTM hybrid (positive = DHBO better)\n",
               imp.round(1).to_markdown(), "\n"]
        # Table 5: Diebold-Mariano
        dm = pd.DataFrame({c: {k: v for k, v in runs[(t, c)]["dm"].items()} for c in cs})

        def star(x):
            s, p = x["DM"], x["p"]
            mark = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            return f"{s:.2f}{mark} (p={p:.3f})"
        dm_s = dm.apply(lambda col: col.map(star))
        dm_s.index.name = "DHBO vs"
        dm_s.to_csv(tdir / f"table5_{t}_dm.csv")
        md += [f"## Table 5-{t}. Diebold-Mariano (HLN) tests, DHBO vs each baseline; negative = DHBO more accurate\n",
               dm_s.to_markdown(), "\n*** p<0.01, ** p<0.05, * p<0.1\n"]
        # Table 6: intervals
        rows = []
        for c in cs:
            for name, v in runs[(t, c)]["intervals"].items():
                rows.append({"Country": c, "Method": name, "Coverage": round(v["coverage"], 3),
                             "Mean width": round(v["width"], 1), "Winkler": round(v["winkler"], 1)})
        iv = pd.DataFrame(rows).set_index("Country")
        iv.to_csv(tdir / f"table6_{t}_intervals.csv")
        md += [f"## Table 6-{t}. Conformal prediction intervals of DHBO (nominal 80% and 95%)\n", iv.to_markdown(), "\n"]
        # Table 7: ablation
        abl_names = list(runs[(t, cs[0])]["metrics_ablation"])
        ab = pd.DataFrame({c: {k: runs[(t, c)]["metrics_ablation"][k]["MAPE"] for k in abl_names} for c in cs})
        ab["Mean"] = ab.mean(axis=1)
        ab.index.name = "Variant (MAPE %)"
        ab.to_csv(tdir / f"table7_{t}_ablation.csv")
        md += [f"## Table 7-{t}. Ablation study, MAPE (%)\n", md_table(ab, list(ab.columns)), "\n"]
        # Table 8: hyper-parameters and stacking weights
        rows = []
        for c in cs:
            r = runs[(t, c)]
            w = np.mean(r["stack_weights"], axis=0)
            rows.append({"Country": c, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r["hp_dhbo"].items()},
                         "w SARIMA": round(w[0], 2), "w ETS": round(w[1], 2), "w Hybrid": round(w[2], 2)})
        hp = pd.DataFrame(rows).set_index("Country")
        hp.to_csv(tdir / f"table8_{t}_hyperparameters.csv")
        md += [f"## Table 8-{t}. Hyper-parameters chosen by Bayesian optimisation and mean stacking weights\n",
               hp.to_markdown(), "\n"]
        figures(out, runs, t, cs, fdir)

    (out / "paper_tables.md").write_text("\n".join(md))
    print(f"tables -> {tdir}\nfigures -> {fdir}\nmarkdown -> {out / 'paper_tables.md'}")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(out, runs, t, cs, fdir):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    # Fig: test-period forecasts with 95% ACI band (H-step-ahead)
    fig, axes = plt.subplots(len(cs), 1, figsize=(7.2, 2.3 * len(cs)), squeeze=False)
    for ax, c in zip(axes[:, 0], cs):
        z = np.load(out / "runs" / f"{t}_{c}.npz")
        H = runs[(t, c)]["horizon"]
        od = pd.to_datetime(z["origin_dates"])
        tgt = od + pd.Timedelta(days=H)
        ax.fill_between(tgt, z["band__ACI 95%__lo"][:, -1], z["band__ACI 95%__hi"][:, -1],
                        color=C_DHBO, alpha=0.15, linewidth=0, label="DHBO 95% interval (ACI)")
        ax.plot(tgt, z["y_true"][:, -1], color=INK, lw=1.6, label="Observed")
        ax.plot(tgt, z["pred__ARIMA-LSTM hybrid"][:, -1], color=C_JAIN, lw=1.4, ls="--", label="ARIMA-LSTM hybrid")
        ax.plot(tgt, z["pred__DHBO (proposed)"][:, -1], color=C_DHBO, lw=1.8, label="DHBO (proposed)")
        ax.set_title(f"{c}: {H}-day-ahead forecasts, test period", loc="left", fontsize=9, color=INK)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(fdir / f"fig_forecasts_{t}.png", dpi=300)
    plt.close(fig)

    # Fig: MAE by horizon, DHBO vs the two strongest baselines in each country
    fig, axes = plt.subplots(1, len(cs), figsize=(7.2, 2.4), squeeze=False)
    for ax, c in zip(axes[0], cs):
        hm = runs[(t, c)]["horizon_mae"]
        h = np.arange(1, len(hm["DHBO (proposed)"]) + 1)
        for k in MODEL_ORDER:
            if k in ("DHBO (proposed)", "ARIMA-LSTM hybrid"):
                continue
            ax.plot(h, hm[k], color=C_OTHER, lw=0.9)
        ax.plot(h, hm["ARIMA-LSTM hybrid"], color=C_JAIN, lw=1.6, ls="--")
        ax.plot(h, hm["DHBO (proposed)"], color=C_DHBO, lw=2.0, marker="o", ms=3)
        ax.set_title(c, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Horizon (days)", fontsize=8, color=INK2)
        ax.set_xticks(h)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        _style(ax)
    axes[0, 0].set_ylabel("MAE (cases)", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=C_DHBO, lw=2), Line2D([], [], color=C_JAIN, lw=1.6, ls="--"),
                Line2D([], [], color=C_OTHER, lw=1)], ["DHBO (proposed)", "ARIMA-LSTM hybrid", "Other baselines"],
               frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fdir / f"fig_horizon_mae_{t}.png", dpi=300)
    plt.close(fig)

    # Fig: average temporal attention weights over the look-back window
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, c in enumerate(cs):
        a = runs[(t, c)]["attention"]
        if a is None:
            continue
        lag = np.arange(-len(a) + 1, 1)
        ax.plot(lag, a, lw=1.6, color=["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"][i], label=c)
    ax.set_xlabel("Day relative to forecast origin", fontsize=8, color=INK2)
    ax.set_ylabel("Mean attention weight", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fdir / f"fig_attention_{t}.png", dpi=300)
    plt.close(fig)

    # Fig: example causal VMD of one window (first country)
    z = np.load(out / "runs" / f"{t}_{cs[0]}.npz")
    modes, w = z["example_modes"], z["example_window"]
    fig, axes = plt.subplots(len(modes) + 1, 1, figsize=(5.2, 1.0 * (len(modes) + 1)), sharex=True)
    axes[0].plot(w - w.mean(), color=INK, lw=1.2)
    axes[0].set_ylabel("signal", fontsize=7, color=INK2)
    for k, ax in enumerate(axes[1:]):
        ax.plot(modes[k], color=C_DHBO, lw=1.1)
        ax.set_ylabel(f"IMF {k + 1}", fontsize=7, color=INK2)
    for ax in axes:
        _style(ax)
    axes[-1].set_xlabel("Day in 128-day window (log scale, de-meaned)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(fdir / f"fig_vmd_{t}_{cs[0]}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
