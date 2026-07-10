"""Returns-based factor attribution: OLS + Newey-West, all hand-rolled with numpy.

Models (regressors are Ken French factors, dependent is a monthly excess/active
return series):

    CAPM      : Mkt-RF
    FF3       : Mkt-RF, SMB, HML
    FF5       : Mkt-RF, SMB, HML, RMW, CMA
    Carhart4  : Mkt-RF, SMB, HML, MOM
    Carhart6  : Mkt-RF, SMB, HML, RMW, CMA, MOM

Estimation:
    beta = (X'X)^{-1} X'y                              (X includes an intercept = alpha)
    HAC (Newey-West) covariance with the Bartlett kernel:
        S = Gamma_0 + sum_{l=1}^{L} (1 - l/(L+1)) (Gamma_l + Gamma_l')
        Cov(beta) = (X'X)^{-1} S (X'X)^{-1}
    where Gamma_l = sum_t (x_t e_t)(x_{t-l} e_{t-l})' and the default lag is the
    Newey-West rule of thumb L = floor(4 (T/100)^{2/9}).

Return decomposition (average return attributed to factors):
    mean(y) = alpha + sum_k beta_k * mean(f_k) + mean(residual=0)
    contribution_k = beta_k * mean(f_k)   (annualized x12)

``statsmodels`` is intentionally NOT imported here; the tests validate this module
against it independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .logging_setup import get_logger

logger = get_logger(__name__)

MONTHS_PER_YEAR = 12

FACTOR_SETS: Dict[str, List[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "Carhart4": ["Mkt-RF", "SMB", "HML", "MOM"],
    "Carhart6": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def newey_west_lag(nobs: int) -> int:
    """Newey-West (1994) automatic lag: floor(4 * (T/100)^{2/9})."""
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


@dataclass
class RegressionResult:
    """OLS + Newey-West regression output for one model."""

    model: str
    param_names: List[str]           # ['alpha', factor1, ...]
    params: pd.Series
    se: pd.Series                    # Newey-West HAC standard errors
    tstats: pd.Series
    pvalues: pd.Series
    r2: float
    adj_r2: float
    nobs: int
    nw_lag: int
    resid: pd.Series
    fitted_means: pd.Series          # mean of each regressor (for contribution decomp)
    _factor_names: List[str] = field(default_factory=list)

    @property
    def alpha_monthly(self) -> float:
        return float(self.params["alpha"])

    @property
    def alpha_annual(self) -> float:
        """Arithmetic annualization of the monthly intercept."""
        return self.alpha_monthly * MONTHS_PER_YEAR

    @property
    def alpha_tstat(self) -> float:
        return float(self.tstats["alpha"])

    @property
    def betas(self) -> pd.Series:
        return self.params.drop("alpha")

    def contributions(self) -> pd.Series:
        """Annualized average-return decomposition: alpha + beta_k * mean(f_k).

        The components sum to the annualized mean of the dependent variable.
        """
        comp = {"alpha": self.alpha_annual}
        for f in self._factor_names:
            comp[f] = float(self.params[f] * self.fitted_means[f] * MONTHS_PER_YEAR)
        return pd.Series(comp)


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Solve OLS via the normal equations; return (beta, resid, XtX_inv)."""
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XtX_inv


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficients (Bartlett kernel).

    Matches statsmodels' ``cov_type='HAC'`` with ``use_correction=False``.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]          # rows are x_t * e_t
    S = u.T @ u                     # Gamma_0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        Gamma_l = u[l:].T @ u[:-l]  # sum_t u_t u_{t-l}'
        S += w * (Gamma_l + Gamma_l.T)
    return XtX_inv @ S @ XtX_inv


def regress(
    y: pd.Series,
    factors: pd.DataFrame,
    model: str,
    nw_lag: Optional[int] = None,
) -> RegressionResult:
    """Regress ``y`` on the factor set named by ``model`` with NW standard errors."""
    if model not in FACTOR_SETS:
        raise KeyError(f"Unknown model {model!r}; choose from {list(FACTOR_SETS)}.")
    factor_names = FACTOR_SETS[model]

    data = pd.concat([y.rename("y"), factors[factor_names]], axis=1).dropna()
    if len(data) <= len(factor_names) + 1:
        raise ValueError(f"Not enough observations ({len(data)}) for model {model}.")

    yv = data["y"].to_numpy()
    Xf = data[factor_names].to_numpy()
    X = np.column_stack([np.ones(len(yv)), Xf])  # intercept first
    n = len(yv)

    beta, resid, _ = _ols_fit(X, yv)
    lag = nw_lag if nw_lag is not None else newey_west_lag(n)
    cov = _newey_west_cov(X, resid, lag)
    se = np.sqrt(np.diag(cov))

    # Goodness of fit.
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    k = X.shape[1]
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    param_names = ["alpha"] + factor_names
    params = pd.Series(beta, index=param_names)
    se_s = pd.Series(se, index=param_names)
    tstats = params / se_s
    # Two-sided p-value from the normal approximation (large-sample HAC inference).
    from math import erf, sqrt
    pvalues = tstats.apply(lambda t: 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))))

    fitted_means = pd.Series(Xf.mean(axis=0), index=factor_names)

    logger.info(
        "%-9s | n=%d lag=%d | alpha(ann)=%+.2f%% (t=%.2f) | R2=%.3f",
        model, n, lag, params["alpha"] * MONTHS_PER_YEAR * 100,
        tstats["alpha"], r2,
    )

    return RegressionResult(
        model=model,
        param_names=param_names,
        params=params,
        se=se_s,
        tstats=tstats,
        pvalues=pvalues,
        r2=r2,
        adj_r2=adj_r2,
        nobs=n,
        nw_lag=lag,
        resid=pd.Series(resid, index=data.index, name="resid"),
        fitted_means=fitted_means,
        _factor_names=factor_names,
    )


def regress_all_models(y: pd.Series, factors: pd.DataFrame) -> Dict[str, RegressionResult]:
    """Fit every model in FACTOR_SETS to ``y``."""
    return {name: regress(y, factors, name) for name in FACTOR_SETS}


def rolling_betas(
    y: pd.Series,
    factors: pd.DataFrame,
    model: str = "Carhart6",
    window: int = 36,
) -> pd.DataFrame:
    """Rolling-window OLS betas (default 36 months) to visualize style drift."""
    factor_names = FACTOR_SETS[model]
    data = pd.concat([y.rename("y"), factors[factor_names]], axis=1).dropna()
    rows = {}
    for end in range(window, len(data) + 1):
        chunk = data.iloc[end - window:end]
        yv = chunk["y"].to_numpy()
        X = np.column_stack([np.ones(window), chunk[factor_names].to_numpy()])
        beta, _, _ = _ols_fit(X, yv)
        rows[chunk.index[-1]] = pd.Series(beta[1:], index=factor_names)
    out = pd.DataFrame(rows).T
    out.index.name = "date"
    return out


def build_excess_return(total_return: pd.Series, rf: pd.Series) -> pd.Series:
    """Excess return = total return - risk-free, aligned on the common index."""
    df = pd.concat([total_return.rename("r"), rf.rename("rf")], axis=1).dropna()
    return (df["r"] - df["rf"]).rename(f"{total_return.name}_excess")
