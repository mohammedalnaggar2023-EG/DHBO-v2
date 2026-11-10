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
| `extra_baselines.py` | N-HiTS and the statistics-only online ensemble |
| `make_tables.py` | Paper tables |
| `DHBO_colab.ipynb` | Runs the whole study in Google Colab |
| `robustness.py` | Robustness checks of Section 3.8: statistics-only online ensemble (`stat`), five single seeds (`seeds`), equally tuned ARIMA–LSTM (`arimalstm`) |
| `nhits.py` | N-HiTS baseline under the same protocol |
| `make_truncated_data.py`, `run_period.sh` | Earlier test period (data ending 31 December 2021) |
| `make_figures.py`, `figure_data.py` | Figures 1–5 and Supplementary Figure S1 |
| `requirements.txt` | Fixed library versions |
| `requirements_extra.txt` | Extra packages for `nhits.py` and `make_figures.py` |
| `results/` | All result files used in the paper |

## How to run

```bash
pip install -r requirements.txt

# 1. Development countries (India, Brazil, Russia, USA)
python run_experiments.py --out results                          # baselines + DHBO-v1
python run_v2.py --v1 results --out results_v2                   # DHBO-v2 (TCN)
python run_v2.py --v1 results --corrector cnn --out results_cnn  # DHBO-v2-CNN

# 2. Held-out countries (same code and settings)
python run_experiments.py --countries France Germany Japan "South Korea" --out val_v1
python run_v2.py --countries France Germany Japan "South Korea" --v1 val_v1 --out val_tcn
python run_v2.py --countries France Germany Japan "South Korea" --v1 val_v1 --corrector cnn --out val_cnn

# 3. Three-design comparison
python compare_designs.py results_v2 results_cnn India,Brazil,Russia,USA results
python compare_designs.py val_tcn val_cnn "France,Germany,Japan,South Korea" val_v1
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
