"""
DHBO-v2: SARIMA + two neural error-correctors (attention BiLSTM and TCN) +
online convex stacking over {SARIMA, ETS, Naive, SARIMA+BiLSTM, SARIMA+TCN} +
adaptive conformal intervals.  VMD is optional (TPE may choose K = 0).

Baseline and DHBO-v1 test predictions are loaded unchanged from the v1 results
folder (--v1), so every comparison uses identical origins and baselines.

    python run_v2.py --data data/confirmed_global.csv --v1 results_final --out results_v2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import dhbo as D
from run_experiments import Experiment

SEGS = ("val_es", "val_cal", "test")


class ExperimentV2(Experiment):

    def spec_bilstm(self, K):
        return dict(channels=["z", "innov_sarima"], modes=K > 0, base="sarima", attention=True, bi=True, cell="lstm")

    def spec_tcn(self, K):
        return dict(channels=["z", "innov_sarima"], modes=K > 0, base="sarima", attention=False, bi=False, cell=self.args.corrector)

    def _objective_mae(self, spec, hp):
        out, _ = self.fit_predict(spec, hp, seeds=[0], segs=("val_es",))
        y_true = D.to_level(self.Y["val_es"], self.target)
        return float(np.mean(np.abs(y_true - D.to_level(out["val_es"], self.target))))

    def tune_bilstm(self, n_trials):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            hp = {"L": trial.suggest_categorical("L", list(self.Lch)),
                  "K": trial.suggest_categorical("K", [0, 3, 4, 5, 6]),
                  "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
                  "layers": trial.suggest_int("layers", 1, 2),
                  "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                  "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                  "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
                  "batch": trial.suggest_categorical("batch", [16, 32, 64])}
            return self._objective_mae(self.spec_bilstm(hp["K"]), hp)

        t0 = time.time()
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(objective, n_trials=n_trials)
        self.log["time_tune_bilstm_s"] = round(time.time() - t0, 1)
        return dict(st.best_params)

    def tune_tcn(self, n_trials, L, K):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            hp = {"L": L, "K": K, "dropout": trial.suggest_float("dropout", 0.0, 0.3)}
            if self.args.corrector == "tcn":
                hp.update(channels=trial.suggest_categorical("channels", [16, 32, 64]),
                          levels=trial.suggest_int("levels", 2, 4),
                          kernel=trial.suggest_categorical("kernel", [2, 3]))
            else:  # standard 1-D CNN
                hp.update(filters=trial.suggest_categorical("filters", [16, 32, 64]),
                          kernel=trial.suggest_categorical("kernel", [2, 3, 5]),
                          conv_layers=trial.suggest_int("conv_layers", 1, 2),
                          dense=trial.suggest_categorical("dense", [32, 64]))
            hp.update({
                  "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                  "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
                  "batch": trial.suggest_categorical("batch", [16, 32, 64])})
            return self._objective_mae(self.spec_tcn(K), hp)

        t0 = time.time()
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(objective, n_trials=n_trials)
        self.log["time_tune_tcn_s"] = round(time.time() - t0, 1)
        return {**st.best_params, "L": L, "K": K}

    def run_v2(self, v1_dir: Path):
        a = self.args
        seeds = list(range(a.seeds))
        print(f"\n=== {self.country} / {self.target}: test origins {len(self.O['test'])} ({self.device})", flush=True)
        self.statistical()
        self.decompose()
        B = self.BASE

        hp_b = self.tune_bilstm(a.trials)
        print(f"   BiLSTM hp {hp_b}", flush=True)
        hp_t = self.tune_tcn(a.tcn_trials, hp_b["L"], hp_b["K"])
        print(f"   {self.args.corrector.upper()} hp {hp_t}", flush=True)
        t0 = time.time()
        bil, att = self.fit_predict(self.spec_bilstm(hp_b["K"]), hp_b, seeds, segs=SEGS, want_att=True)
        tcn, _ = self.fit_predict(self.spec_tcn(hp_t["K"]), hp_t, seeds, segs=SEGS)
        self.log["time_final_training_s"] = round(time.time() - t0, 1)

        COR = self.args.corrector.upper()
        NET2 = f"SARIMA+{COR}"
        V2 = "DHBO-v2 (proposed)" if self.args.corrector == "tcn" else "DHBO-v2-CNN"
        cat = lambda d: np.concatenate([d[g] for g in SEGS])
        Ycat = cat(self.Y)
        Ocat = np.concatenate([self.O[g] for g in SEGS])
        n_es, n_cal = len(self.O["val_es"]), len(self.O["val_cal"])
        pool = {"SARIMA": cat(B["sarima"]), "ETS": cat(B["ets"]), "Naive": cat(B["naive"]),
                "SARIMA+BiLSTM": cat(bil), NET2: cat(tcn)}

        def online(names, window=a.window):
            out, W = D.stack_online([pool[k] for k in names], Ycat, Ocat, start=n_es, window=window)
            return out[:n_cal], out[n_cal:], W            # val_cal part, test part, weights

        full = list(pool)
        cal_v2, test_v2, W_v2 = online(full)
        variants = {
            "DHBO-v2 (full)": test_v2,
            f"w/o {COR} corrector": online(["SARIMA", "ETS", "Naive", "SARIMA+BiLSTM"])[1],
            "w/o BiLSTM corrector": online(["SARIMA", "ETS", "Naive", NET2])[1],
            "w/o Naive in pool": online(["SARIMA", "ETS", "SARIMA+BiLSTM", NET2])[1],
            "Static stacking (v1-style)": D.stack_predict([pool[k][n_es + n_cal:] for k in full],
                                                          D.stack_fit([pool[k][n_es:n_es + n_cal] for k in full], self.Y["val_cal"])),
            "Online window 30": online(full, 30)[1],
            "Online window 90": online(full, 90)[1],
            "SARIMA+BiLSTM only": pool["SARIMA+BiLSTM"][n_es + n_cal:],
            f"{NET2} only": pool[NET2][n_es + n_cal:],
        }

        # v1 and baselines: identical predictions to the published results
        v1 = np.load(v1_dir / "runs" / f"{self.target}_{self.country}.npz")
        y_true = D.to_level(self.Y["test"], self.target)
        assert np.allclose(v1["y_true"], y_true), "test targets differ from v1 run"
        assert list(v1["origin_dates"]) == [str(self.dates[t].date()) for t in self.O["test"]]
        models = {k[6:]: v1[k] for k in v1.files if k.startswith("pred__")}
        models[V2] = D.to_level(test_v2, self.target)
        metrics = {k: D.point_metrics(y_true, v, self.mase_scale) for k, v in models.items()}
        metrics_var = {k: D.point_metrics(y_true, D.to_level(v, self.target), self.mase_scale) for k, v in variants.items()}
        horizon_mae = {k: D.per_horizon_mae(y_true, v).tolist() for k, v in models.items()}

        e2 = np.mean(np.abs(y_true - models[V2]), axis=1)
        dm = {}
        for k, v in models.items():
            if k != V2:
                s, p = D.dm_test(e2, np.mean(np.abs(y_true - v), axis=1), self.H)
                dm[k] = {"DM": s, "p": p}

        cal_scores = np.abs(self.Y["val_cal"] - cal_v2)
        intervals, bands = {}, {}
        for alpha in (0.2, 0.05):
            for adaptive in (False, True):
                lo, hi = D.conformal_intervals(test_v2, self.Y["test"], self.O["test"], cal_scores,
                                               alpha=alpha, adaptive=adaptive)
                lo, hi = D.to_level(lo, self.target), D.to_level(hi, self.target)
                name = f"{'ACI' if adaptive else 'Split CP'} {int(100 * (1 - alpha))}%"
                intervals[name] = D.interval_scores(y_true, lo, hi, alpha)
                bands[name] = (lo, hi)

        W_test = W_v2[n_cal:]
        res = dict(country=self.country, target=self.target, horizon=self.H, n_origins={g: int(len(o)) for g, o in self.O.items()},
                   hp_bilstm=hp_b, hp_tcn=hp_t, corrector=self.args.corrector, model_key=V2, pool=full, window=a.window,
                   mean_weights_test=W_test.mean(axis=(0, 1)).tolist(),
                   mean_weights_test_by_h=W_test.mean(axis=0).tolist(),
                   metrics=metrics, metrics_variants=metrics_var, horizon_mae=horizon_mae, dm=dm,
                   intervals=intervals, attention=(att.tolist() if att is not None else None),
                   sarima_order=str(self.sarima_order), log=self.log, seeds=seeds,
                   trials=a.trials, tcn_trials=a.tcn_trials)
        d = self.out / "runs"
        d.mkdir(parents=True, exist_ok=True)
        tag = f"{self.target}_{self.country}"
        (d / f"{tag}.json").write_text(json.dumps(res, indent=2))
        np.savez_compressed(d / f"{tag}.npz", origin_dates=v1["origin_dates"], y_true=y_true,
                            **{f"pred__{k}": v for k, v in models.items()},
                            **{f"band__{k}__lo": v[0] for k, v in bands.items()},
                            **{f"band__{k}__hi": v[1] for k, v in bands.items()},
                            weights_test=W_test)
        m2, m1 = metrics[V2], metrics["DHBO (proposed)"]
        print(f"   MASE v2 {m2['MASE']:.3f} | v1 {m1['MASE']:.3f} | hybrid {metrics['ARIMA-LSTM hybrid']['MASE']:.3f} "
              f"| ACI95 {intervals['ACI 95%']['coverage']:.3f} | weights {np.round(res['mean_weights_test'], 2)}", flush=True)
        return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=D.JHU_URL)
    p.add_argument("--v1", default="results_final")
    p.add_argument("--targets", nargs="+", default=["daily", "cumulative"])
    p.add_argument("--countries", nargs="+", default=list(D.COUNTRIES))
    p.add_argument("--dengue-data", default=D.DENGUE_FILE)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--vmd-window", type=int, default=None)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--tcn-trials", type=int, default=15)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--window", type=int, default=60, help="online stacking window (origins)")
    p.add_argument("--corrector", choices=["tcn", "cnn"], default="tcn", help="second neural corrector")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--out", default="results_v2")
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    if a.fast:
        a.trials, a.tcn_trials, a.seeds, a.max_epochs, a.patience = 4, 3, 1, 40, 8
    import torch
    torch.set_num_threads(int(os.environ.get("DHBO_THREADS", max(1, os.cpu_count() or 1))))
    out = Path(a.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    t0 = time.time()
    for target in a.targets:
        df = D.load_data(target, a.dengue_data if target == "weekly" else a.data)
        for country in a.countries:
            ExperimentV2(df, country, target, a, out).run_v2(Path(a.v1))
    print(f"\nall runs done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
