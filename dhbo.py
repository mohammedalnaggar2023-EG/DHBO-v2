"""
DHBO: Decomposition-based Hybrid SARIMA - Attention BiLSTM with
Bayesian Optimization and Conformal uncertainty for epidemic forecasting.

Library module: data, decomposition, statistical models, neural models,
stacking, conformal prediction, metrics and statistical tests.

All forecasts are made from rolling origins: a forecast issued at origin t
uses only observations up to and including t (no look-ahead).  The VMD
decomposition is applied causally on a sliding window that ends at t, so the
decomposition never sees the future either.
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

JHU_URL = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
           "csse_covid_19_data/csse_covid_19_time_series/"
           "time_series_covid19_confirmed_global.csv")

# display name -> JHU "Country/Region" name (same four countries as Jain et al., 2024)
COUNTRIES = {"India": "India", "Brazil": "Brazil", "Russia": "Russia", "USA": "US"}
# External validation set, fixed by a rule before any result was seen: the four
# countries with the most cumulative cases in the JHU file, excluding those above.
VALIDATION_COUNTRIES = {"France": "France", "Germany": "Germany", "Japan": "Japan", "South Korea": "Korea, South"}
ALL_COUNTRIES = {**COUNTRIES, **VALIDATION_COUNTRIES}

# Second disease: weekly dengue cases (OpenDengue V1.3, national temporal extract).
# Fixed by a rule before any result was seen: in each WHO region with series of at
# least 10 years (520 weeks) of contiguous weekly national reports, the two countries
# with the highest mean weekly cases.
DENGUE_COUNTRIES = {"Brazil": "BRAZIL", "Mexico": "MEXICO", "Malaysia": "MALAYSIA",
                    "Laos": "LAO PEOPLE'S DEMOCRATIC REPUBLIC"}
DENGUE_FILE = "data/dengue_weekly.csv"

# Frequency-dependent settings: seasonal period, horizon, look-back choices, VMD window
FREQ = {"daily": dict(s=7, H=7, L=(14, 21, 28), W=128),
        "cumulative": dict(s=7, H=7, L=(14, 21, 28), W=128),
        "weekly": dict(s=52, H=4, L=(26, 39, 52), W=104)}


def load_dengue(path: str = DENGUE_FILE) -> pd.DataFrame:
    """Weekly dengue cases, one row per country (display name), columns = week start."""
    d = pd.read_csv(path, parse_dates=["week_start"])
    return d.pivot(index="country", columns="week_start", values="cases").astype(float)


def load_data(target: str, path: str | None = None) -> pd.DataFrame:
    if target == "weekly":
        return load_dengue(path or DENGUE_FILE)
    return load_jhu(path or JHU_URL)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_jhu(path_or_url: str = JHU_URL) -> pd.DataFrame:
    """Load the JHU CSSE global confirmed-cases file, one row per country."""
    df = pd.read_csv(path_or_url)
    df = df.drop(columns=["Province/State", "Lat", "Long"]).groupby("Country/Region").sum()
    df.columns = pd.to_datetime(df.columns, format="%m/%d/%y")
    return df.astype(float)


def build_series(df: pd.DataFrame, country: str, target: str,
                 min_cum: float = 100.0, smooth: int = 7) -> pd.Series:
    """Return the modelled series for one country.

    target = "cumulative": cumulative confirmed cases (the Jain et al. setting).
    target = "daily": 7-day moving average of daily new cases (harder, more useful).
    Downward revisions in the cumulative count are removed with a running maximum,
    and the series starts on the first day with at least `min_cum` cases.
    """
    if target == "weekly":  # dengue: weekly counts on the contiguous weekly record
        return df.loc[country].dropna().rename(f"{country}_weekly")
    cum = df.loc[ALL_COUNTRIES.get(country, country)].cummax()
    cum = cum[cum >= min_cum]
    if target == "cumulative":
        return cum.rename(f"{country}_cumulative")
    if target == "daily":
        daily = cum.diff().clip(lower=0)
        daily = daily.rolling(smooth, min_periods=smooth).mean().dropna()
        return daily.rename(f"{country}_daily")
    raise ValueError(target)


@dataclass
class Split:
    """Chronological split: train | val_es | val_cal | test (index boundaries)."""
    n: int
    a: int  # end of train (exclusive)
    b: int  # end of val_es
    c: int  # end of val_cal; test = [c, n)

    @classmethod
    def make(cls, n, train=0.6, val_es=0.1, val_cal=0.1):
        return cls(n, int(n * train), int(n * (train + val_es)), int(n * (train + val_es + val_cal)))

    def origins(self, segment: str, H: int, t_min: int) -> np.ndarray:
        """Origins t whose whole target window t+1..t+H lies inside the segment."""
        lo, hi = {"train": (0, self.a), "val_es": (self.a, self.b),
                  "val_cal": (self.b, self.c), "test": (self.c, self.n)}[segment]
        start = max(lo - 1, t_min)
        stop = hi - H  # last valid t satisfies t + H <= hi - 1
        return np.arange(start, stop, dtype=int)


def target_matrix(z: np.ndarray, origins: np.ndarray, H: int) -> np.ndarray:
    """Matrix of future values z[t+1..t+H] for each origin t, shape (N, H)."""
    return np.stack([z[t + 1:t + 1 + H] for t in origins])


# ---------------------------------------------------------------------------
# 2. Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014)
#    Self-contained numpy implementation following the reference algorithm.
# ---------------------------------------------------------------------------
def vmd(f: np.ndarray, K: int, alpha: float = 2000.0, tau: float = 0.0,
        tol: float = 1e-7, max_iter: int = 500):
    """Decompose signal f (even length) into K band-limited modes.

    Returns (modes, omega): modes has shape (K, len(f)), sorted from the lowest
    to the highest centre frequency; omega holds the centre frequencies.
    """
    f = np.asarray(f, dtype=float)
    if len(f) % 2:
        raise ValueError("VMD window length must be even")
    half = len(f) // 2
    f_mirror = np.concatenate([f[:half][::-1], f, f[-half:][::-1]])  # mirror extension
    T = len(f_mirror)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1.0 / T
    f_hat_plus = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus[:T // 2] = 0  # keep the positive spectrum only

    omega = np.arange(K) * (0.5 / K)
    u_hat = np.zeros((T, K), dtype=complex)
    lam = np.zeros(T, dtype=complex)
    pos = slice(T // 2, T)
    for _ in range(max_iter):
        u_prev = u_hat.copy()
        for k in range(K):  # Gauss-Seidel sweep over modes
            others = u_hat.sum(axis=1) - u_hat[:, k]
            u_hat[:, k] = (f_hat_plus - others - lam / 2) / (1 + alpha * (freqs - omega[k]) ** 2)
            power = np.abs(u_hat[pos, k]) ** 2
            omega[k] = np.dot(freqs[pos], power) / max(power.sum(), 1e-300)
        lam = lam + tau * (u_hat.sum(axis=1) - f_hat_plus)
        if np.sum(np.abs(u_hat - u_prev) ** 2) / T < tol:
            break

    # rebuild the full (Hermitian) spectrum and return to the time domain
    full = np.zeros((T, K), dtype=complex)
    full[pos, :] = u_hat[pos, :]
    idx = np.arange(1, T // 2 + 1)[::-1]
    full[idx, :] = np.conj(u_hat[pos, :])
    full[0, :] = np.conj(full[-1, :])
    modes = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=0), axis=0)).T
    modes = modes[:, T // 4:3 * T // 4]
    order = np.argsort(omega)
    return modes[order], omega[order]


def _vmd_window(args):
    w, K, Lmax, alpha = args
    modes, _ = vmd(w - w.mean(), K, alpha=alpha)
    return modes[:, -Lmax:].T  # (Lmax, K): last Lmax values of each mode


def causal_vmd_features(z: np.ndarray, K: int, W: int, Lmax: int,
                        alpha: float = 2000.0, n_jobs: int = -1) -> np.ndarray:
    """For every origin t >= W-1, decompose only z[t-W+1..t] and keep the last
    Lmax samples of each mode.  Returns array (n, Lmax, K), NaN where undefined.
    Sliding-window decomposition avoids the look-ahead leakage that occurs when
    VMD is applied once to the whole series."""
    from joblib import Parallel, delayed
    n = len(z)
    out = np.full((n, Lmax, K), np.nan)
    ts = list(range(W - 1, n))
    res = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(_vmd_window)((z[t - W + 1:t + 1], K, Lmax, alpha)) for t in ts)
    for t, r in zip(ts, res):
        out[t] = r
    return out


# ---------------------------------------------------------------------------
# 3. Statistical components
# ---------------------------------------------------------------------------
def select_d(z_train: np.ndarray, max_d: int = 2, alpha: float = 0.05) -> int:
    """Order of differencing from repeated ADF tests (at least 1 for epidemic levels)."""
    from statsmodels.tsa.stattools import adfuller
    d, x = 0, np.asarray(z_train, float)
    while d < max_d:
        try:
            p = adfuller(x, autolag="AIC")[1]
        except Exception:
            p = 1.0
        if p < alpha:
            break
        x, d = np.diff(x), d + 1
    return max(d, 1)


def fit_sarima(z_train: np.ndarray, seasonal: bool = True, s: int = 7,
               max_p: int = 2, max_q: int = 2):
    """AIC grid search over (p,d,q)(P,0,Q)_s on the training data only."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    d = select_d(z_train)
    PQ = [(0, 0), (1, 0), (0, 1), (1, 1)] if seasonal else [(0, 0)]
    best, best_aic, best_order = None, np.inf, None
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        for P, Q in PQ:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m = SARIMAX(z_train, order=(p, d, q),
                                seasonal_order=(P, 0, Q, s) if (P or Q) else (0, 0, 0, 0))
                    r = m.fit(disp=False, maxiter=200)
                except Exception:
                    continue
            if np.isfinite(r.aic) and r.aic < best_aic:
                best, best_aic, best_order = r, r.aic, ((p, d, q), (P, 0, Q, s) if (P or Q) else None)
    if best is None:
        raise RuntimeError("SARIMA grid search failed")
    return best, best_order


def sarima_rolling(res, z_full: np.ndarray, origins: np.ndarray, H: int):
    """H-step forecasts from each origin with parameters fixed at their training
    estimates.  The Kalman filter runs forward only, so the forecast from origin t
    uses z[0..t].  Also returns the one-step-ahead innovation series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = res.apply(z_full)
        out = np.empty((len(origins), H))
        for i, t in enumerate(origins):
            out[i] = np.asarray(full.get_prediction(start=t + 1, end=t + H, dynamic=True).predicted_mean)
        innov = np.asarray(full.resid, dtype=float)
    return out, innov


class DampedHolt:
    """Additive damped-trend exponential smoothing (ETS(A,Ad,N)), fitted by SSE.
    Implemented directly so it can be run from rolling origins with fixed params."""

    def fit(self, z_train):
        z = np.asarray(z_train, float)

        def sse(p):
            return np.sum(self._run(z, *p)[2] ** 2)

        x0 = [0.5, 0.1, 0.95]
        bounds = [(1e-3, 0.999), (1e-4, 0.999), (0.80, 0.995)]
        self.params = optimize.minimize(sse, x0, bounds=bounds, method="L-BFGS-B").x
        return self

    @staticmethod
    def _run(z, a, b, phi):
        n = len(z)
        lev, tr, err = np.empty(n), np.empty(n), np.zeros(n)
        lev[0], tr[0] = z[0], z[1] - z[0] if n > 1 else 0.0
        for t in range(1, n):
            pred = lev[t - 1] + phi * tr[t - 1]
            err[t] = z[t] - pred
            lev[t] = pred + a * err[t]
            tr[t] = phi * tr[t - 1] + a * b * err[t]
        return lev, tr, err

    def rolling(self, z_full, origins, H):
        a, b, phi = self.params
        lev, tr, _ = self._run(np.asarray(z_full, float), a, b, phi)
        damp = np.cumsum(phi ** np.arange(1, H + 1))
        return np.stack([lev[t] + damp * tr[t] for t in origins])


def naive_rolling(z_full, origins, H):
    return np.repeat(z_full[origins][:, None], H, axis=1)


# ---------------------------------------------------------------------------
# 4. Neural components (PyTorch)
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def build_net(n_in, H, hidden=64, layers=1, dropout=0.2, attention=True,
              bidirectional=True, cell="lstm"):
    torch = _torch()
    nn = torch.nn

    class RecurrentForecaster(nn.Module):
        """(Bi)LSTM/GRU encoder + optional additive (Bahdanau) temporal attention
        + MLP head that emits all H horizons at once (direct multi-output)."""

        def __init__(self):
            super().__init__()
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn(n_in, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
            d = hidden * (2 if bidirectional else 1)
            self.attention = attention
            if attention:
                self.att_W = nn.Linear(d, d)
                self.att_v = nn.Linear(d, 1, bias=False)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(d * (2 if attention else 1), d), nn.ReLU(), nn.Linear(d, H))
            self.last_attention = None

        def forward(self, x):
            h, _ = self.rnn(x)                                   # (B, L, d)
            last = h[:, -1, :]
            if self.attention:
                score = self.att_v(torch.tanh(self.att_W(h))).squeeze(-1)  # (B, L)
                a = torch.softmax(score, dim=1)
                ctx = (a.unsqueeze(-1) * h).sum(dim=1)
                self.last_attention = a.detach()
                feat = torch.cat([ctx, last], dim=-1)
            else:
                feat = last
            return self.head(self.drop(feat))

    return RecurrentForecaster()


def build_tcn(n_in, H, channels=32, levels=3, kernel=3, dropout=0.1):
    """Temporal Convolutional Network (Bai et al., 2018): stacked residual blocks of
    dilated causal 1-D convolutions (dilation 1, 2, 4, ...), weight-normalised, with
    ReLU and dropout.  The receptive field is 1 + 2 (kernel - 1)(2^levels - 1) steps.
    The output at the last time step feeds a linear head with H outputs."""
    torch = _torch()
    nn = torch.nn
    wn = nn.utils.parametrizations.weight_norm

    class CausalConv(nn.Module):
        def __init__(self, cin, cout, dil):
            super().__init__()
            self.pad = (kernel - 1) * dil
            self.conv = wn(nn.Conv1d(cin, cout, kernel, dilation=dil))

        def forward(self, x):                      # x: (B, C, L); pad on the left only = causal
            return self.conv(nn.functional.pad(x, (self.pad, 0)))

    class Block(nn.Module):
        def __init__(self, cin, cout, dil):
            super().__init__()
            self.net = nn.Sequential(CausalConv(cin, cout, dil), nn.ReLU(), nn.Dropout(dropout),
                                     CausalConv(cout, cout, dil), nn.ReLU(), nn.Dropout(dropout))
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            return torch.relu(self.net(x) + self.skip(x))

    class TCNForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(*[Block(n_in if i == 0 else channels, channels, 2 ** i) for i in range(levels)])
            self.head = nn.Linear(channels, H)
            self.last_attention = None

        def forward(self, x):                      # x: (B, L, C)
            h = self.blocks(x.transpose(1, 2))     # (B, channels, L)
            return self.head(h[:, :, -1])

    return TCNForecaster()


def build_cnn(n_in, H, L, filters=32, kernel=3, conv_layers=2, dense=64, dropout=0.1):
    """Standard 1-D CNN forecaster: stacked 'same'-padded convolutions with ReLU,
    max-pooling over time, then a dense layer and a linear head with H outputs.
    Unlike the TCN it has no dilation, no causal padding inside the window and no
    residual connections; the window itself lies entirely in the past."""
    torch = _torch()
    nn = torch.nn

    class CNNForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            layers, cin = [], n_in
            for _ in range(conv_layers):
                layers += [nn.Conv1d(cin, filters, kernel, padding="same"), nn.ReLU()]
                cin = filters
            layers += [nn.MaxPool1d(2)]
            self.conv = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters * (L // 2), dense), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(dense, H))
            self.last_attention = None

        def forward(self, x):                      # x: (B, L, C)
            return self.head(self.conv(x.transpose(1, 2)))

    return CNNForecaster()


@dataclass
class Scaler:
    mx: np.ndarray = None
    sx: np.ndarray = None
    my: float = 0.0
    sy: float = 1.0

    def fit(self, X, Y):
        self.mx = X.reshape(-1, X.shape[-1]).mean(0)
        self.sx = X.reshape(-1, X.shape[-1]).std(0) + 1e-8
        self.my, self.sy = float(Y.mean()), float(Y.std() + 1e-8)
        return self

    def x(self, X):
        return (X - self.mx) / self.sx

    def y(self, Y):
        return (Y - self.my) / self.sy

    def y_inv(self, Y):
        return Y * self.sy + self.my


def train_net(Xtr, Ytr, Xva, Yva, H, hp: dict, seed: int = 0, attention=True,
              bidirectional=True, cell="lstm", max_epochs=200, patience=20, device="cpu"):
    """Train with AdamW + Huber loss, early stopping on the val_es loss.
    Returns (model, scaler, best_val_loss)."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = Scaler().fit(Xtr, Ytr)
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    xtr, ytr, xva, yva = to(sc.x(Xtr)), to(sc.y(Ytr)), to(sc.x(Xva)), to(sc.y(Yva))
    if cell == "tcn":
        net = build_tcn(Xtr.shape[-1], H, hp["channels"], hp["levels"], hp["kernel"], hp["dropout"]).to(device)
    elif cell == "cnn":
        net = build_cnn(Xtr.shape[-1], H, Xtr.shape[1], hp["filters"], hp["kernel"], hp["conv_layers"],
                        hp["dense"], hp["dropout"]).to(device)
    else:
        net = build_net(Xtr.shape[-1], H, hp["hidden"], hp["layers"], hp["dropout"],
                        attention, bidirectional, cell).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, wait, bs = np.inf, None, 0, int(hp.get("batch", 32))
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(xva), yva).item()
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, sc, best


def predict_net(net, sc: Scaler, X, device="cpu", return_attention=False):
    torch = _torch()
    with torch.no_grad():
        out = net(torch.tensor(sc.x(X), dtype=torch.float32, device=device))
        yhat = sc.y_inv(out.cpu().numpy())
        att = net.last_attention.cpu().numpy() if (return_attention and net.last_attention is not None) else None
    return (yhat, att) if return_attention else yhat


def window_inputs(origins, L, channels: list[np.ndarray], modes: np.ndarray | None = None,
                  level: np.ndarray | None = None):
    """Stack input sequences of length L ending at each origin.

    channels: full-length 1-D series (e.g. log level, SARIMA innovations).
    If `level` is given, it is subtracted from each window's value at origin t
    (level normalisation).  `modes` is the (n, Lmax, K) causal VMD array; only
    its last L rows for origin t are used."""
    seqs = []
    for t in origins:
        cols = []
        for ch in channels:
            w = ch[t - L + 1:t + 1]
            if level is not None and ch is level:
                w = w - level[t]
            cols.append(w[:, None])
        if modes is not None:
            cols.append(modes[t, -L:, :])
        seqs.append(np.concatenate(cols, axis=1))
    return np.stack(seqs).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Stacking meta-learner (convex combination per horizon)
# ---------------------------------------------------------------------------
def fit_convex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P: (N, M) base forecasts for one horizon, y: (N,). Weights >= 0, sum to 1,
    minimising squared error (log scale)."""
    M = P.shape[1]
    # Because the weights sum to 1, P @ w - y = E @ w with E = P - y.  Errors on the
    # log scale are tiny (squared error ~1e-6), below SLSQP's default stopping
    # tolerance, which made the optimiser stop at its starting point (equal
    # weights).  Standardising E leaves the minimiser unchanged and fixes that.
    E = P - y[:, None]
    E = E / (E.std() + 1e-12)
    obj = lambda w: np.mean((E @ w) ** 2)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    r = optimize.minimize(obj, np.full(M, 1.0 / M), bounds=[(0, 1)] * M,
                          constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    if not r.success:
        warnings.warn(f"stacking optimiser did not converge: {r.message}; using equal weights")
        return np.full(M, 1.0 / M)
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def stack_fit(bases_cal: list[np.ndarray], y_cal: np.ndarray) -> np.ndarray:
    """Per-horizon convex weights. bases_cal: list of (N, H). Returns (H, M)."""
    H = y_cal.shape[1]
    return np.stack([fit_convex_weights(np.stack([b[:, h] for b in bases_cal], 1), y_cal[:, h])
                     for h in range(H)])


def stack_predict(bases: list[np.ndarray], W: np.ndarray) -> np.ndarray:
    P = np.stack(bases, axis=-1)  # (N, H, M)
    return np.einsum("nhm,hm->nh", P, W)


def stack_crossfit(bases_cal, y_cal):
    """Out-of-sample stacked predictions on the calibration block (2-fold, by time)
    so that conformal calibration scores are not in-sample."""
    N = len(y_cal)
    m = N // 2
    out = np.empty_like(y_cal)
    W2 = stack_fit([b[:m] for b in bases_cal], y_cal[:m])
    out[m:] = stack_predict([b[m:] for b in bases_cal], W2)
    W1 = stack_fit([b[m:] for b in bases_cal], y_cal[m:])
    out[:m] = stack_predict([b[:m] for b in bases_cal], W1)
    return out


def stack_online(bases: list[np.ndarray], y: np.ndarray, origins: np.ndarray,
                 start: int, window: int = 60, min_pairs: int = 20, init_W: np.ndarray | None = None):
    """Online convex stacking with delayed feedback.

    bases: list of (N, H) base forecasts over consecutive origins (time order);
    y: (N, H) truths; origins: (N,) origin indices t.  For every row i >= start and
    horizon h, the weights are refitted on the most recent `window` rows j whose
    target t_j + h + 1 is already observed at origin t_i (t_j + h + 1 <= t_i).
    Rows before `start` are history only.  Returns stacked forecasts for rows
    start..N-1 and the weights used, shape (N - start, H, M)."""
    N, H = y.shape
    M = len(bases)
    P = np.stack(bases, axis=-1)                       # (N, H, M)
    out = np.empty((N - start, H))
    Ws = np.empty((N - start, H, M))
    for h in range(H):
        for i in range(start, N):
            avail = np.where(origins[:i] + h + 1 <= origins[i])[0]
            avail = avail[-window:]
            if len(avail) >= min_pairs:
                w = fit_convex_weights(P[avail, h, :], y[avail, h])
            else:
                w = init_W[h] if init_W is not None else np.full(M, 1.0 / M)
            Ws[i - start, h] = w
            out[i - start, h] = P[i, h, :] @ w
    return out, Ws


# ---------------------------------------------------------------------------
# 6. Conformal prediction intervals (log scale, per horizon)
# ---------------------------------------------------------------------------
def _conformal_q(scores, alpha):
    s = np.sort(np.asarray(scores))
    m = len(s)
    k = int(np.ceil((m + 1) * (1 - alpha)))
    return np.inf if k > m else s[k - 1]


def conformal_intervals(pred_test, y_test, origins_test, cal_scores, alpha=0.05,
                        adaptive=True, gamma=0.005, window=200):
    """Split conformal (adaptive=False) or Adaptive Conformal Inference
    (Gibbs & Candes, 2021; adaptive=True) with *delayed* feedback: the score of a
    forecast issued at origin t for step h only becomes available at time t+h.

    pred_test, y_test: (N, H) on the log scale; cal_scores: (Nc, H) absolute
    residuals from the calibration block.  Returns lower, upper (N, H)."""
    N, H = pred_test.shape
    lo, hi = np.empty((N, H)), np.empty((N, H))
    for h in range(H):
        scores = list(cal_scores[:, h])
        a_t = alpha
        pending = []  # (time_realised, score, lower, upper, truth)
        for i, t in enumerate(origins_test):
            still = []
            for (tr, sc_, l_, u_, y_) in pending:
                if tr <= t:  # truth observed by time t
                    scores.append(sc_)
                    if adaptive:
                        err = float(not (l_ <= y_ <= u_))
                        a_t = float(np.clip(a_t + gamma * (alpha - err), 1e-4, 0.999))
                else:
                    still.append((tr, sc_, l_, u_, y_))
            pending = still
            base = scores[-window:] if adaptive else list(cal_scores[:, h])
            q = _conformal_q(base, a_t if adaptive else alpha)
            if not np.isfinite(q):
                q = max(base)
            lo[i, h], hi[i, h] = pred_test[i, h] - q, pred_test[i, h] + q
            pending.append((t + h + 1, abs(y_test[i, h] - pred_test[i, h]), lo[i, h], hi[i, h], y_test[i, h]))
    return lo, hi


def interval_scores(y, lo, hi, alpha):
    """Coverage, mean width and Winkler score in the original scale."""
    cover = np.mean((y >= lo) & (y <= hi))
    width = np.mean(hi - lo)
    pen = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return {"coverage": float(cover), "width": float(width), "winkler": float(np.mean(hi - lo + pen))}


# ---------------------------------------------------------------------------
# 7. Point metrics and the Diebold-Mariano test
# ---------------------------------------------------------------------------
def point_metrics(y, yhat, mase_scale):
    """y, yhat: (N, H) in the original scale. Averaged over origins and horizons."""
    e = y - yhat
    nz = y != 0
    ape = np.abs(e[nz] / y[nz]) * 100
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAPE": float(np.mean(ape)),
        "sMAPE": float(np.mean(200 * np.abs(e) / (np.abs(y) + np.abs(yhat) + 1e-12))),
        "MdAPE": float(np.median(ape)),
        "MASE": float(np.mean(np.abs(e)) / mase_scale),
    }


def per_horizon_mae(y, yhat):
    return np.mean(np.abs(y - yhat), axis=0)


def dm_test(err1, err2, h, power=1):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
    correction.  err*: (N,) forecast errors per origin (here: mean absolute error
    over the H horizons).  Negative statistic -> model 1 more accurate."""
    d = np.abs(err1) ** power - np.abs(err2) ** power
    n = len(d)
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        var = gam[0] / n
    stat = dbar / math.sqrt(var)
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def to_level(zhat, target):
    """Back-transform from log1p scale; daily counts cannot be negative."""
    y = np.expm1(zhat)
    return np.clip(y, 0, None)
