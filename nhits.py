"""N-HiTS baseline under the same protocol as DHBO (Section 3.8).

Usage:  python nhits.py          # development countries
        python nhits.py held     # also the four held-out countries
"""
import os
CODE = os.environ.get('DHBO_CODE', '.')                      # folder with dhbo.py, run_experiments.py, run_v2.py
RESULTS = os.environ.get('DHBO_RESULTS', 'results')          # published result folders
DATA = os.environ.get('DHBO_DATA', 'data/confirmed_global.csv')  # JHU CSSE confirmed-case file
OUT = os.environ.get('DHBO_OUT', '.')                         # where outputs are written
import sys, json, time, warnings, logging
warnings.filterwarnings('ignore'); logging.getLogger('pytorch_lightning').setLevel(logging.ERROR); logging.getLogger('lightning').setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, CODE); import dhbo as D
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
RES = Path(RESULTS); DEV = ['India','Brazil','Russia','USA']; HELD = ['France','Germany','Japan','South Korea']
H = 7
df = D.load_jhu(DATA)
countries = DEV + (HELD if 'held' in sys.argv else [])
out = {}
for target in ['daily', 'cumulative']:
    for c in countries:
        s = D.build_series(df, c, target); y = s.values.astype(float); z = np.log1p(y); n = len(z)
        sp = D.Split.make(n); O = sp.origins('test', H, 127); sc = float(np.mean(np.abs(np.diff(y[:sp.a]))))
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        tr = pd.DataFrame({'unique_id': 'x', 'ds': dates[:sp.b], 'y': z[:sp.b]})
        ctx = pd.concat([pd.DataFrame({'unique_id': f'o{t}', 'ds': dates[t-127:t+1], 'y': z[t-127:t+1]}) for t in O])
        t0 = time.time(); preds = []
        for seed in (0, 1, 2):
            m = NHITS(h=H, input_size=28, max_steps=300, mlp_units=3 * [[128, 128]], early_stop_patience_steps=10, val_check_steps=20,
                      scaler_type='robust', random_seed=seed, enable_progress_bar=False, enable_model_summary=False, logger=False)
            nf = NeuralForecast(models=[m], freq='D'); nf.fit(df=tr, val_size=sp.b - sp.a)
            f = nf.predict(df=ctx)
            f = f.reset_index() if 'unique_id' not in f.columns else f
            preds.append(np.stack([f[f.unique_id == f'o{t}']['NHITS'].values[:H] for t in O]))
        P = D.to_level(np.mean(preds, 0), target)
        vdir = RES / (f'results_v2_{target}' if c in DEV else f'val_tcn_{target}')
        v2 = np.load(vdir / 'runs' / f'{target}_{c}.npz'); yt = v2['y_true']
        assert len(yt) == len(P)
        k2 = [k for k in v2.files if k.startswith('pred__DHBO-v2')][0]
        stat, p = D.dm_test(np.mean(np.abs(yt - v2[k2]), 1), np.mean(np.abs(yt - P), 1), H)
        out[f'{target}|{c}'] = dict(MASE_nhits=float(np.mean(np.abs(yt - P)) / sc), MASE_v2=float(np.mean(np.abs(yt - v2[k2])) / sc),
                                    MASE_sarima=float(np.mean(np.abs(yt - v2['pred__SARIMA'])) / sc), DM_v2_vs_nhits=stat, p=p, time_s=round(time.time() - t0))
        print(target, c, {k: round(v, 3) for k, v in out[f'{target}|{c}'].items()}, flush=True)
        json.dump(out, open(os.path.join(OUT, 'nhits.json'), 'w'), indent=1)
