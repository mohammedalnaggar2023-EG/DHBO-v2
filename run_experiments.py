"""
Run the full DHBO study: proposed model, baselines, ablations, conformal
intervals and Diebold-Mariano tests, for every country and target.

Usage (Colab or laptop):
    pip install -r requirements.txt
    python run_experiments.py                    # full study (several hours on CPU)
    python run_experiments.py --fast             # quick end-to-end check (~10-20 min)
    python run_experiments.py --data covid.csv   # use a local copy of the JHU file

Outputs go to ./results: per-run JSON/NPZ files, tables (CSV + Markdown) and
figures (PNG).  results/paper_tables.md holds every table ready for the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import dhbo as D

DEFAULT_HP = {"L": 21, "K": 4, "hidden": 64, "layers": 1, "dropout": 0.2,
              "lr": 1e-3, "wd": 1e-5, "batch": 32}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=D.JHU_URL, help="path or URL of the JHU confirmed_global CSV")
    p.add_argument("--targets", nargs="+", default=["daily", "cumulative"])
    p.add_argument("--countries", nargs="+", default=list(D.COUNTRIES))
    p.add_argument("--horizon", type=int, default=None, help="forecast horizon in days (default 7)")
    p.add_argument("--vmd-window", type=int, default=None, help="VMD window in days (default 128)")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials for DHBO")
    p.add_argument("--baseline-trials", type=int, default=15, help="Optuna trials for LSTM/GRU baselines")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--out", default="results")
    p.add_argument("--fast", action="store_true", help="tiny budget for a quick end-to-end check")
    a = p.parse_args()
    if a.fast:
        a.trials, a.baseline_trials, a.seeds, a.max_epochs, a.patience = 6, 3, 1, 60, 10
    return a


class Experiment:
    """Everything for one (country, target) pair."""

    def __init__(self, df, country, target, args, out: Path):
        import torch
        self.args, self.country, self.target, self.out = args, country, target, out
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        s = D.build_series(df, country, target)
        self.dates = s.index
        self.y = s.values.astype(float)
        self.z = np.log1p(self.y)
        self.n = len(self.z)
        f = D.FREQ[target]
        self.s, self.Lch = f["s"], f["L"]
        self.H = args.horizon or f["H"]
        self.W = args.vmd_window or f["W"]
        self.sp = D.Split.make(self.n)
        self.Lmax = max(self.Lch)
        t_min = self.W - 1
        self.O = {g: self.sp.origins(g, self.H, t_min) for g in ("train", "val_es", "val_cal", "test")}
        self.Y = {g: D.target_matrix(self.z, o, self.H) for g, o in self.O.items()}
        self.mase_scale = float(np.mean(np.abs(np.diff(self.y[:self.sp.a]))))
        self.log = {}

    # ---------------- statistical components ----------------
    def statistical(self):
        a = self.sp.a
        t0 = time.time()
        sres, self.sarima_order = D.fit_sarima(self.z[:a], seasonal=True, s=self.s)
        ares, self.arima_order = D.fit_sarima(self.z[:a], seasonal=False)
        ets = D.DampedHolt().fit(self.z[:a])
        self.BASE = {"sarima": {}, "arima": {}, "ets": {}, "naive": {}}
        for g, o in self.O.items():
            self.BASE["sarima"][g], innov_s = D.sarima_rolling(sres, self.z, o, self.H)
            self.BASE["arima"][g], innov_a = D.sarima_rolling(ares, self.z, o, self.H)
            self.BASE["ets"][g] = ets.rolling(self.z, o, self.H)
            self.BASE["naive"][g] = D.naive_rolling(self.z, o, self.H)
        self.CH = {"z": self.z, "innov_sarima": innov_s, "innov_arima": innov_a}
        self.log["ets_params"] = [float(v) for v in ets.params]
        self.log["time_statistical_s"] = round(time.time() - t0, 1)
        print(f"   SARIMA {self.sarima_order}  ARIMA {self.arima_order[0]}  ({self.log['time_statistical_s']} s)")

    # ---------------- causal VMD features ----------------
    def decompose(self, Ks=(3, 4, 5, 6)):
        cache = self.out / "cache"
        cache.mkdir(exist_ok=True)
        self.MODES = {}
        t0 = time.time()
        for K in Ks:
            f = cache / f"vmd_{self.country}_{self.target}_K{K}_W{self.W}.npy"
            if f.exists():
                self.MODES[K] = np.load(f)
            else:
                self.MODES[K] = D.causal_vmd_features(self.z, K, self.W, self.Lmax)
                np.save(f, self.MODES[K])
        self.log["time_vmd_s"] = round(time.time() - t0, 1)
        # one full window decomposition for the figure (window ending at the last training day)
        t = self.sp.a - 1
        w = self.z[t - self.W + 1:t + 1]
        self.example_modes, self.example_omega = D.vmd(w - w.mean(), 4)
        self.example_window = w

    # ---------------- neural pipelines ----------------
    def make_xy(self, spec, g, L, K):
        o = self.O[g]
        chans = [self.CH[c] for c in spec["channels"]]
        level = self.CH["z"] if "z" in spec["channels"] else None
        modes = self.MODES[K] if spec["modes"] else None
        X = D.window_inputs(o, L, chans, modes, level=level)
        base = self.BASE[spec["base"]][g] if spec["base"] != "level" else D.naive_rolling(self.z, o, self.H)
        return X, self.Y[g] - base, base

    def fit_predict(self, spec, hp, seeds, segs=("val_cal", "test"), want_att=False):
        K = hp.get("K", 4)
        Xtr, Rtr, _ = self.make_xy(spec, "train", hp["L"], K)
        Xva, Rva, _ = self.make_xy(spec, "val_es", hp["L"], K)
        preds = {g: [] for g in segs}
        atts = []
        for sd in seeds:
            net, sc, _ = D.train_net(Xtr, Rtr, Xva, Rva, self.H, hp, seed=sd,
                                     attention=spec["attention"], bidirectional=spec["bi"],
                                     cell=spec["cell"], max_epochs=self.args.max_epochs,
                                     patience=self.args.patience, device=self.device)
            for g in segs:
                X, _, base = self.make_xy(spec, g, hp["L"], K)
                if want_att and g == "test":
                    r, att = D.predict_net(net, sc, X, self.device, return_attention=True)
                    atts.append(att.mean(0))
                else:
                    r = D.predict_net(net, sc, X, self.device)
                preds[g].append(base + r)
        out = {g: np.mean(v, axis=0) for g, v in preds.items()}
        return out, (np.mean(atts, axis=0) if atts else None)

    def tune(self, spec, n_trials, name):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        y_true = D.to_level(self.Y["val_es"], self.target)

        def objective(trial):
            hp = {"L": trial.suggest_categorical("L", list(self.Lch)),
                  "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
                  "layers": trial.suggest_int("layers", 1, 2),
                  "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                  "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                  "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
                  "batch": trial.suggest_categorical("batch", [16, 32, 64])}
            if spec["modes"]:
                hp["K"] = trial.suggest_categorical("K", [3, 4, 5, 6])
            out, _ = self.fit_predict(spec, hp, seeds=[0], segs=("val_es",))
            return float(np.mean(np.abs(y_true - D.to_level(out["val_es"], self.target))))

        t0 = time.time()
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=n_trials)
        hp = dict(study.best_params)
        hp.setdefault("K", 4)
        self.log[f"time_tune_{name}_s"] = round(time.time() - t0, 1)
        self.log[f"best_val_mae_{name}"] = float(study.best_value)
        return hp

    # ---------------- stacking + conformal ----------------
    def stack(self, base_names_or_preds):
        cal = [b["val_cal"] for b in base_names_or_preds]
        tst = [b["test"] for b in base_names_or_preds]
        W = D.stack_fit(cal, self.Y["val_cal"])
        return D.stack_predict(tst, W), D.stack_crossfit(cal, self.Y["val_cal"]), W

    # ---------------- full run ----------------
    def run(self):
        a = self.args
        seeds = list(range(a.seeds))
        print(f"\n=== {self.country} / {self.target}: n={self.n}, "
              f"test {self.dates[self.sp.c].date()} -> {self.dates[-1].date()} "
              f"({len(self.O['test'])} origins, H={self.H}) on {self.device}")
        self.statistical()
        self.decompose()
        B = self.BASE

        S = {  # neural pipeline specifications
            "dhbo": dict(channels=["z", "innov_sarima"], modes=True, base="sarima", attention=True, bi=True, cell="lstm"),
            "no_vmd": dict(channels=["z", "innov_sarima"], modes=False, base="sarima", attention=True, bi=True, cell="lstm"),
            "no_att": dict(channels=["z", "innov_sarima"], modes=True, base="sarima", attention=False, bi=True, cell="lstm"),
            "no_sarima": dict(channels=["z"], modes=True, base="level", attention=True, bi=True, cell="lstm"),
            "lstm": dict(channels=["z"], modes=False, base="level", attention=False, bi=False, cell="lstm"),
            "gru": dict(channels=["z"], modes=False, base="level", attention=False, bi=False, cell="gru"),
            "arima_lstm": dict(channels=["innov_arima"], modes=False, base="arima", attention=False, bi=False, cell="lstm"),
        }

        print("   tuning DHBO ...")
        hp_dhbo = self.tune(S["dhbo"], a.trials, "dhbo")
        print(f"   best DHBO hp {hp_dhbo}")
        hp_lstm = self.tune(S["lstm"], a.baseline_trials, "lstm")
        hp_gru = self.tune(S["gru"], a.baseline_trials, "gru")
        hp_jain = {**DEFAULT_HP, "L": self.Lch[0]}  # fixed, as a typical ARIMA-LSTM hybrid

        print("   final training ...")
        hyb, att = self.fit_predict(S["dhbo"], hp_dhbo, seeds, want_att=True)
        nv, _ = self.fit_predict(S["no_vmd"], hp_dhbo, seeds)
        na, _ = self.fit_predict(S["no_att"], hp_dhbo, seeds)
        nb, _ = self.fit_predict(S["dhbo"], {**DEFAULT_HP, "L": self.Lch[1]}, seeds)
        ns, _ = self.fit_predict(S["no_sarima"], hp_dhbo, seeds)
        lstm, _ = self.fit_predict(S["lstm"], hp_lstm, seeds)
        gru, _ = self.fit_predict(S["gru"], hp_gru, seeds)
        jain, _ = self.fit_predict(S["arima_lstm"], hp_jain, seeds)

        sar, ets = B["sarima"], B["ets"]
        dhbo_test, dhbo_cal_oos, W = self.stack([sar, ets, hyb])
        abl = {
            "DHBO (full)": dhbo_test,
            "w/o VMD decomposition": self.stack([sar, ets, nv])[0],
            "w/o attention": self.stack([sar, ets, na])[0],
            "w/o Bayesian optimisation": self.stack([sar, ets, nb])[0],
            "w/o stacking (SARIMA + residual BiLSTM)": hyb["test"],
            "w/o SARIMA (VMD + Att-BiLSTM only)": self.stack([ets, ns])[0],
        }
        models = {
            "Naive": B["naive"]["test"],
            "ARIMA": B["arima"]["test"],
            "SARIMA": sar["test"],
            "ETS (damped)": ets["test"],
            "LSTM": lstm["test"],
            "GRU": gru["test"],
            "ARIMA-LSTM hybrid": jain["test"],
            "DHBO (proposed)": dhbo_test,
        }

        y_true = D.to_level(self.Y["test"], self.target)
        lvl = {k: D.to_level(v, self.target) for k, v in models.items()}
        lvl_abl = {k: D.to_level(v, self.target) for k, v in abl.items()}
        metrics = {k: D.point_metrics(y_true, v, self.mase_scale) for k, v in lvl.items()}
        metrics_abl = {k: D.point_metrics(y_true, v, self.mase_scale) for k, v in lvl_abl.items()}
        horizon_mae = {k: D.per_horizon_mae(y_true, v).tolist() for k, v in lvl.items()}

        e_dhbo = np.mean(np.abs(y_true - lvl["DHBO (proposed)"]), axis=1)
        dm = {}
        for k, v in lvl.items():
            if k == "DHBO (proposed)":
                continue
            stat, p = D.dm_test(e_dhbo, np.mean(np.abs(y_true - v), axis=1), self.H)
            dm[k] = {"DM": stat, "p": p}

        cal_scores = np.abs(self.Y["val_cal"] - dhbo_cal_oos)
        intervals, bands = {}, {}
        for alpha in (0.2, 0.05):
            for adaptive in (False, True):
                lo, hi = D.conformal_intervals(dhbo_test, self.Y["test"], self.O["test"], cal_scores,
                                               alpha=alpha, adaptive=adaptive)
                lo, hi = D.to_level(lo, self.target), D.to_level(hi, self.target)
                name = f"{'ACI' if adaptive else 'Split CP'} {int(100 * (1 - alpha))}%"
                intervals[name] = D.interval_scores(y_true, lo, hi, alpha)
                bands[name] = (lo, hi)

        res = dict(country=self.country, target=self.target, n=self.n, horizon=self.H,
                   dates=dict(start=str(self.dates[0].date()), train_end=str(self.dates[self.sp.a - 1].date()),
                              val_es_end=str(self.dates[self.sp.b - 1].date()),
                              val_cal_end=str(self.dates[self.sp.c - 1].date()), test_start=str(self.dates[self.sp.c].date()),
                              end=str(self.dates[-1].date())),
                   n_origins={g: int(len(o)) for g, o in self.O.items()},
                   sarima_order=str(self.sarima_order), arima_order=str(self.arima_order[0]),
                   hp_dhbo=hp_dhbo, hp_lstm=hp_lstm, hp_gru=hp_gru, hp_arima_lstm=hp_jain,
                   stack_weights=W.tolist(), stack_base_names=["SARIMA", "ETS", "SARIMA+Att-BiLSTM"],
                   metrics=metrics, metrics_ablation=metrics_abl, horizon_mae=horizon_mae,
                   dm=dm, intervals=intervals, attention=(att.tolist() if att is not None else None),
                   log=self.log, seeds=seeds, trials=a.trials, baseline_trials=a.baseline_trials)
        d = self.out / "runs"
        d.mkdir(exist_ok=True)
        tag = f"{self.target}_{self.country}"
        (d / f"{tag}.json").write_text(json.dumps(res, indent=2))
        origin_dates = np.array([str(self.dates[t].date()) for t in self.O["test"]])
        np.savez_compressed(d / f"{tag}.npz", origin_dates=origin_dates, y_true=y_true,
                            dates=np.array([str(x.date()) for x in self.dates]), y=self.y,
                            **{f"pred__{k}": v for k, v in lvl.items()},
                            **{f"band__{k}__lo": v[0] for k, v in bands.items()},
                            **{f"band__{k}__hi": v[1] for k, v in bands.items()},
                            example_window=self.example_window, example_modes=self.example_modes,
                            example_omega=self.example_omega)
        print(f"   DHBO MAPE {metrics['DHBO (proposed)']['MAPE']:.2f}%  |  "
              f"ARIMA-LSTM {metrics['ARIMA-LSTM hybrid']['MAPE']:.2f}%  |  "
              f"ACI-95 coverage {intervals['ACI 95%']['coverage']:.3f}")
        return res


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("DHBO_THREADS", max(1, os.cpu_count() or 1))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print("loading data ...")
    df = D.load_jhu(args.data)
    t0 = time.time()
    for target in args.targets:
        for country in args.countries:
            Experiment(df, country, target, args, out).run()
    print(f"\nall runs done in {(time.time() - t0) / 60:.1f} min")
    import make_tables
    make_tables.main(out)


if __name__ == "__main__":
    main()
