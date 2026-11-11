# DHBO-v2: Adaptive SARIMA–BiLSTM–TCN Ensemble for Short-Term Epidemic Forecasting

Code and result files for the paper:

> M. K. Elnagar, T. E. Taha, F. E. Abd El-Samie, A. S. El-Fishawy,
> "Learning When to Trust the Correction: An Adaptive SARIMA–BiLSTM–TCN Ensemble with
> Bayesian Tuning and Conformal Intervals for Short-Term Epidemic Forecasting" (under review).

Department of Electronics and Electrical Communications Engineering, Faculty of Electronic
Engineering, Menoufia University, Menouf, Egypt.

## What DHBO-v2 does

For every forecast origin *t*, DHBO-v2 predicts days *t*+1 … *t*+7 in five steps:

1. **SARIMA** gives a baseline forecast on the log(1 + y) scale.
2. **Causal VMD** (optional) decomposes only the window that ends at *t*, so no future data leak in.
3. An **attention BiLSTM** and a **TCN** each learn a correction to the SARIMA errors
   (hyper-parameters tuned with Optuna's TPE).
4. **Online stacking** blends SARIMA, damped ETS, the naive forecast and the two hybrids
   with convex weights re-estimated at every origin from errors already observed.
5. **Adaptive conformal inference (ACI)** adds prediction intervals, with delayed feedback
   for multi-step outcomes.

## Data

The study uses the public COVID-19 time series from the Center for Systems Science and
Engineering (CSSE) at Johns Hopkins University:

- Repository: https://github.com/CSSEGISandData/COVID-19
- File: `csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv`

Download that file and save it as `data/confirmed_global.csv` before running the scripts.
The data are not included in this repository.

## Files

| File | Purpose |
|---|---|
| `dhbo.py` | Core library: SARIMA, causal VMD, attention BiLSTM, TCN, CNN, stacking, conformal intervals, metrics and tests |
| `run_experiments.py` | Baselines (Naive, ARIMA, SARIMA, damped ETS, LSTM, GRU, ARIMA–LSTM) and DHBO-v1 |
| `run_v2.py` | DHBO-v2 (TCN corrector, default) or DHBO-v2-CNN (`--corrector cnn`), on the same origins |
| `analyze_v2.py` | Summaries and tables for DHBO-v2 |
| `compare_designs.py` | Comparison of DHBO-v1, DHBO-v2 and DHBO-v2-CNN |
| `make_tables.py` | Paper tables |
| `DHBO_colab.ipynb` | Runs the whole study in Google Colab |
| `robustness.py` | Robustness checks of Section 3.8: statistics-only online ensemble (`stat`), five single seeds (`seeds`), equally tuned ARIMA–LSTM (`arimalstm`) |
| `nhits.py` | N-HiTS baseline under the same protocol |
| `make_truncated_data.py`, `run_period.sh` | Earlier test period (data ending 31 December 2021) |
| `make_figures.py`, `figure_data.py` | Figures 1–5 and Supplementary Figure S1 |
| `requirements.txt` | Fixed library versions |
| `requirements_extra.txt` | Extra packages for `nhits.py` and `make_figures.py` |
| `results/` | All result files used in the paper (see below) |
| `.gitignore` | Files Git should not track (raw data, caches, logs) |

### Result files (`results/`)

| Folder | Contents |
|---|---|
| `results_final/` | Baselines and DHBO-v1, development countries, daily and cumulative (`runs/`, `tables/`, `figures/`, `paper_tables.md`) |
| `results_v2_daily/`, `results_v2_cumulative/` | DHBO-v2 (TCN), development countries |
| `results_cnn_daily/`, `results_cnn_cumulative/` | DHBO-v2-CNN, development countries |
| `val_v1_daily/`, `val_v1_cumulative/` | Baselines and DHBO-v1, held-out countries |
| `val_tcn_daily/`, `val_tcn_cumulative/` | DHBO-v2 (TCN), held-out countries |
| `val_cnn_daily/`, `val_cnn_cumulative/` | DHBO-v2-CNN, held-out countries |
| `compare_*.txt`, `compare_*.json` | Output of `compare_designs.py` |

Each `runs/<target>_<country>.json` holds all metrics, Diebold–Mariano tests, intervals, hyper-parameters,
stacking weights and dates; each `.npz` holds the test-period forecasts and interval bands of every model.

## How to run

```bash
pip install -r requirements.txt

# 1. Development countries (India, Brazil, Russia, USA)
python run_experiments.py --data data/confirmed_global.csv --out run_v1                       # baselines + DHBO-v1
python run_v2.py --data data/confirmed_global.csv --v1 run_v1 --out run_v2                    # DHBO-v2 (TCN)
python run_v2.py --data data/confirmed_global.csv --v1 run_v1 --corrector cnn --out run_cnn   # DHBO-v2-CNN
python analyze_v2.py 'run_v2/runs/*.json'                                                     # DHBO-v2 summaries

# 2. Held-out countries (same code and settings)
python run_experiments.py --data data/confirmed_global.csv --countries France Germany Japan "South Korea" --out run_val_v1
python run_v2.py --data data/confirmed_global.csv --countries France Germany Japan "South Korea" --v1 run_val_v1 --out run_val_tcn
python run_v2.py --data data/confirmed_global.csv --countries France Germany Japan "South Korea" --v1 run_val_v1 --corrector cnn --out run_val_cnn

# 3. Three-design comparison
python compare_designs.py run_v2 run_cnn India,Brazil,Russia,USA run_v1
python compare_designs.py run_val_tcn run_val_cnn "France,Germany,Japan,South Korea" run_val_v1
```

The same summaries can be produced directly from the published result files, without re-running anything:

```bash
python analyze_v2.py                                   # reads results/results_v2_*/runs/*.json
python compare_designs.py results/results_v2 results/results_cnn India,Brazil,Russia,USA results/results_final
python compare_designs.py results/val_tcn results/val_cnn "France,Germany,Japan,South Korea" results/val_v1
```

### Robustness checks and figures

These scripts read the published result folders in `results/` (for example
`results/results_v2_daily/`, `results/val_tcn_daily/`, `results/results_final/`).

```bash
pip install -r requirements_extra.txt

python robustness.py stat        # statistics-only online ensemble
python robustness.py seeds       # seed variability
python robustness.py arimalstm   # ARIMA–LSTM with the same tuning budget
python nhits.py held             # N-HiTS on development and held-out countries
bash run_period.sh               # earlier test period (data ending 31 Dec 2021)
python make_figures.py           # figures/ (PNG and TIFF, 1200 dpi)
```

Outputs: `out_stat/stat_stack.json`, `out_seeds/seeds.json`, `out_arimalstm/arimalstm_tuned.json`,
`nhits.json`, `per_v1_*` and `per_v2_*` (earlier period) and `figures/`.

Paths can be changed with the environment variables `DHBO_DATA`, `DHBO_RESULTS`,
`DHBO_CODE` and `DHBO_OUT`.

Add `--fast` to any command for a quick end-to-end check. The full study takes several
hours on a CPU; the Colab notebook runs every step in order.

Library versions matter: ARIMA order selection changed between statsmodels versions, so
use the versions in `requirements.txt` to reproduce the published numbers.

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`).

## License

MIT License (see `LICENSE`).
