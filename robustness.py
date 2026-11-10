"""Robustness experiments for DHBO-v2 (Section 3.8 of the paper).

Tasks:
  stat       statistics-only online ensemble (SARIMA, damped ETS, naive) vs DHBO-v2
  seeds      DHBO-v2 re-run with five single seeds, and the published three-seed setting
  arimalstm  ARIMA-LSTM hybrid given the same 30-trial tuning budget as DHBO

Usage:  python robustness.py stat|seeds|arimalstm
Uses dhbo.py, run_experiments.py and run_v2.py unchanged; reads the published result files.
"""
import os
CODE = os.environ.get('DHBO_CODE', '.')                      # folder with dhbo.py, run_experiments.py, run_v2.py
RESULTS = os.environ.get('DHBO_RESULTS', 'results')          # published result folders
DATA = os.environ.get('DHBO_DATA', 'data/confirmed_global.csv')  # JHU CSSE confirmed-case file
OUT = os.environ.get('DHBO_OUT', '.')                         # where outputs are written
import sys, json, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, CODE)
import dhbo as D
from run_experiments import Experiment
from run_v2 import ExperimentV2, SEGS
RES = Path(RESULTS)
DEV = ['India', 'Brazil', 'Russia', 'USA']; HELD = ['France', 'Germany', 'Japan', 'South Korea']
def ns(**kw):
    a = dict(horizon=7, vmd_window=128, max_epochs=200, patience=20, trials=30, baseline_trials=15, seeds=3,
             corrector='tcn', window=60, tcn_trials=15, fast=False)
    a.update(kw); return argparse.Namespace(**a)
def v2_dir(country, target):
    return RES / (f'results_v2_{target}' if country in DEV else f'val_tcn_{target}')
def mase(y, p, sc): return float(np.mean(np.abs(y - p)) / sc)

def task_stat(df, out):
    rows = {}
    for target in ['daily', 'cumulative']:
        for c in DEV + HELD:
            ex = Experiment(df, c, target, ns(), out); ex.statistical(); B = ex.BASE
            cat = lambda d: np.concatenate([d[g] for g in SEGS])
            Ycat = cat(ex.Y); Ocat = np.concatenate([ex.O[g] for g in SEGS]); n_es = len(ex.O['val_es']); n_cal = len(ex.O['val_cal'])
            st, W = D.stack_online([cat(B['sarima']), cat(B['ets']), cat(B['naive'])], Ycat, Ocat, start=n_es, window=60)
            st_test = st[n_cal:]
            med = np.median(np.stack([B['sarima']['test'], B['ets']['test'], B['naive']['test']]), 0)
            v2 = np.load(v2_dir(c, target) / 'runs' / f'{target}_{c}.npz')
            y = v2['y_true']; tl = lambda z: D.to_level(z, target)
            assert np.allclose(tl(B['sarima']['test']), v2['pred__SARIMA']), 'SARIMA differs from stored'
            P = {'Stat-stack': tl(st_test), 'Stat-median': tl(med)}
            k2 = [k for k in v2.files if k.startswith('pred__DHBO-v2')][0]
            e2 = np.mean(np.abs(y - v2[k2]), 1)
            r = {'MASE_v2': mase(y, v2[k2], ex.mase_scale), 'MASE_SARIMA': mase(y, v2['pred__SARIMA'], ex.mase_scale)}
            for name, p in P.items():
                r[f'MASE_{name}'] = mase(y, p, ex.mase_scale)
                s, pv = D.dm_test(e2, np.mean(np.abs(y - p), 1), ex.H); r[f'DM_v2_vs_{name}'] = s; r[f'p_v2_vs_{name}'] = pv
            s, pv = D.dm_test(np.mean(np.abs(y - P['Stat-stack']), 1), np.mean(np.abs(y - v2['pred__SARIMA']), 1), ex.H)
            r['DM_stack_vs_SARIMA'] = s; r['p_stack_vs_SARIMA'] = pv
            r['mean_weights'] = W[n_cal:].mean(axis=(0, 1)).tolist()
            rows[f'{target}|{c}'] = r; print(target, c, {k: round(v, 3) for k, v in r.items() if not isinstance(v, list)}, flush=True)
    json.dump(rows, open(out / 'stat_stack.json', 'w'), indent=1)

def task_seeds(df, out, seeds=(0, 1, 2, 3, 4)):
    rows = {}
    for target in ['daily', 'cumulative']:
        for c in DEV:
            j = json.load(open(v2_dir(c, target) / 'runs' / f'{target}_{c}.json'))
            hp_b, hp_t = j['hp_bilstm'], j['hp_tcn']
            ex = ExperimentV2(df, c, target, ns(), out); ex.statistical()
            K = hp_b['K']
            if K > 0: ex.decompose(Ks=(K,))
            else: ex.MODES = {}
            B = ex.BASE; cat = lambda d: np.concatenate([d[g] for g in SEGS])
            Ycat = cat(ex.Y); Ocat = np.concatenate([ex.O[g] for g in SEGS]); n_es = len(ex.O['val_es']); n_cal = len(ex.O['val_cal'])
            v2 = np.load(v2_dir(c, target) / 'runs' / f'{target}_{c}.npz'); y = v2['y_true']
            per = {}; preds = []
            t0 = time.time()
            for sd in seeds:
                bil, _ = ex.fit_predict(ex.spec_bilstm(K), hp_b, [sd], segs=SEGS)
                tcn, _ = ex.fit_predict(ex.spec_tcn(hp_t['K']), hp_t, [sd], segs=SEGS)
                preds.append((bil, tcn))
                pool = [cat(B['sarima']), cat(B['ets']), cat(B['naive']), cat(bil), cat(tcn)]
                st, _ = D.stack_online(pool, Ycat, Ocat, start=n_es, window=60)
                per[sd] = mase(y, D.to_level(st[n_cal:], target), ex.mase_scale)
            # published configuration: each network averaged over seeds 0, 1, 2
            avg = lambda i: {g: np.mean([preds[s][i][g] for s in (0, 1, 2)], 0) for g in SEGS}
            pool3 = [cat(B['sarima']), cat(B['ets']), cat(B['naive']), cat(avg(0)), cat(avg(1))]
            st3, _ = D.stack_online(pool3, Ycat, Ocat, start=n_es, window=60)
            rep = mase(y, D.to_level(st3[n_cal:], target), ex.mase_scale)
            pub = j['metrics']['DHBO-v2 (proposed)']['MASE']
            rows[f'{target}|{c}'] = dict(per_seed=per, mean=float(np.mean(list(per.values()))), sd=float(np.std(list(per.values()), ddof=1)),
                                         reproduced_3seed=rep, published=pub, sarima=j['metrics']['SARIMA']['MASE'],
                                         v1=j['metrics']['DHBO (proposed)']['MASE'], time_s=round(time.time() - t0))
            print(target, c, rows[f'{target}|{c}'], flush=True)
            json.dump(rows, open(out / 'seeds.json', 'w'), indent=1)

def task_arimalstm(df, out, trials=30):
    rows = {}
    spec = dict(channels=["innov_arima"], modes=False, base="arima", attention=False, bi=False, cell="lstm")
    for target in ['daily', 'cumulative']:
        for c in DEV:
            ex = Experiment(df, c, target, ns(), out); ex.statistical()
            t0 = time.time(); hp = ex.tune(spec, trials, 'arima_lstm')
            p, _ = ex.fit_predict(spec, hp, [0, 1, 2], segs=('test',))
            v2 = np.load(v2_dir(c, target) / 'runs' / f'{target}_{c}.npz'); y = v2['y_true']
            pt = D.to_level(p['test'], target); k2 = [k for k in v2.files if k.startswith('pred__DHBO-v2')][0]
            s, pv = D.dm_test(np.mean(np.abs(y - v2[k2]), 1), np.mean(np.abs(y - pt), 1), ex.H)
            rows[f'{target}|{c}'] = dict(hp=hp, MASE_tuned=mase(y, pt, ex.mase_scale), MASE_fixed=mase(y, v2['pred__ARIMA-LSTM hybrid'], ex.mase_scale),
                                         MASE_v2=mase(y, v2[k2], ex.mase_scale), DM_v2_vs_tuned=s, p=pv, time_s=round(time.time() - t0))
            print(target, c, rows[f'{target}|{c}'], flush=True)
            json.dump(rows, open(out / 'arimalstm_tuned.json', 'w'), indent=1)

if __name__ == '__main__':
    import torch, warnings; warnings.filterwarnings('ignore')
    torch.set_num_threads(int(os.environ.get('DHBO_THREADS', 1)))
    task = sys.argv[1]; out = Path(OUT) / f'out_{task}'; (out / 'cache').mkdir(parents=True, exist_ok=True)
    df = D.load_jhu(DATA)
    {'stat': task_stat, 'seeds': task_seeds, 'arimalstm': task_arimalstm}[task](df, out)
