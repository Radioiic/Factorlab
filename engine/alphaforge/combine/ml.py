"""Walk-forward machine-learning factor combination.

:class:`MLCombiner` stacks cross-sectionally ranked factors into a long-format
feature matrix, fits a regressor on the rank of the forward return and
re-fits it every ``refit_every`` dates using only labels that were *fully
realised* at the refit date.  The output is a prediction panel that can be fed
to portfolio construction exactly like a single factor.

Each model trains only on rows whose outcome is known at its use date, and the
model retrains on a fixed schedule.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from alphaforge.ops.cs import cs_rank

logger = logging.getLogger("alphaforge.combine.ml")

__all__ = ["MLCombiner", "MODELS"]

#: Supported model names.
MODELS: tuple[str, ...] = ("hgb", "ridge")

#: Neutral fill for a missing rank feature (ranks live in ``(0, 1]``) - used
#: only by models that cannot handle ``NaN`` natively (Ridge).
_RANK_FILL: float = 0.5

#: Upper bound on the rows used for permutation importance.
_MAX_IMPORTANCE_ROWS: int = 20_000

_HGB_DEFAULTS: dict[str, Any] = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 4}
_RIDGE_DEFAULTS: dict[str, Any] = {"alpha": 1.0}


class MLCombiner:
    """Walk-forward ML combiner over cross-sectionally ranked factors.

    Args:
        train_window: Maximum number of *label-known* dates in each training
            set (the most recent ones are used).
        refit_every: Refit cadence in dates; each model predicts the
            ``refit_every`` dates following its refit date.
        min_train: Minimum number of label-known dates (with at least one
            usable training row) before the first refit.
        model: ``"hgb"`` (``HistGradientBoostingRegressor(max_iter=200,
            learning_rate=0.05, max_depth=4, random_state=seed)``) or
            ``"ridge"`` (``Ridge(alpha=1.0)``).
        seed: Random state for the model and the importance subsample.
        **model_kwargs: Extra keyword arguments forwarded to the estimator,
            overriding the defaults above.

    Attributes:
        feature_importances_: ``pd.Series`` indexed by factor name from the
            *last* fitted model - permutation importance (``n_repeats=3`` on at
            most 20k rows of the last training set) for ``"hgb"``, ``|coef|``
            for ``"ridge"``; ``None`` before any fit.
        refit_dates_: Dates at which a model was fitted.
        refit_history_: One dict per refit (``date``, ``n_rows``,
            ``train_start``, ``train_end``, ``pred_start``, ``pred_end``,
            ``fit_seconds``).
        feature_names_: Factor names in feature order.
        last_model_: The most recently fitted estimator.
    """

    def __init__(
        self,
        *,
        train_window: int = 250,
        refit_every: int = 21,
        min_train: int = 120,
        model: str = "hgb",
        seed: int = 0,
        **model_kwargs: Any,
    ) -> None:
        self.train_window = int(train_window)
        self.refit_every = int(refit_every)
        self.min_train = int(min_train)
        self.model = str(model).lower()
        self.seed = int(seed)
        self.model_kwargs = dict(model_kwargs)
        if self.train_window < 1:
            raise ValueError(f"train_window must be >= 1, got {train_window}")
        if self.refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")
        if self.min_train < 1:
            raise ValueError(f"min_train must be >= 1, got {min_train}")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")

        self.feature_importances_: pd.Series | None = None
        self.refit_dates_: list[pd.Timestamp] = []
        self.refit_history_: list[dict[str, Any]] = []
        self.feature_names_: list[str] = []
        self.last_model_: Any = None

    # ------------------------------------------------------------ internals
    def _make_model(self) -> Any:
        if self.model == "hgb":
            params = {**_HGB_DEFAULTS, "random_state": self.seed, **self.model_kwargs}
            return HistGradientBoostingRegressor(**params)
        params = {**_RIDGE_DEFAULTS, **self.model_kwargs}
        return Ridge(**params)

    def _importances(self, model: Any, x: np.ndarray, y: np.ndarray) -> pd.Series:
        """Feature importances of ``model`` on (a subsample of) its training set."""
        idx = pd.Index(self.feature_names_, name="factor")
        if self.model == "ridge":
            coef = np.abs(np.asarray(model.coef_, dtype=float).ravel())
            return pd.Series(coef, index=idx, name="abs_coef")
        n = x.shape[0]
        if n > _MAX_IMPORTANCE_ROWS:
            rng = np.random.default_rng(self.seed)
            sel = np.sort(rng.choice(n, _MAX_IMPORTANCE_ROWS, replace=False))
            x, y = x[sel], y[sel]
        res = permutation_importance(model, x, y, n_repeats=3, random_state=self.seed)
        return pd.Series(np.asarray(res.importances_mean, dtype=float), index=idx, name="permutation_importance")

    # ------------------------------------------------------------ public API
    def fit_predict(
        self,
        factors: dict[str, pd.DataFrame],
        fwd_ret: pd.DataFrame,
        *,
        horizon: int,
        lag: int,
    ) -> pd.DataFrame:
        """Walk-forward fit / predict.

        Let ``T`` be the position of a refit date.  The label of date ``d``
        (``fwd_ret`` at ``d``) is realised at the close of ``d + lag + horizon``,
        so the training set at ``T`` is the last ``train_window`` dates with
        position ``<= T - lag - horizon`` (rows with a finite label and at
        least one finite feature).  The fitted model predicts positions
        ``T + 1 .. T + refit_every``.  Refits happen at the first date where at
        least ``min_train`` such label-known dates exist and every
        ``refit_every`` dates thereafter.

        Args:
            factors: ``{name: panel}``; each is aligned onto ``fwd_ret``'s grid
                and cross-sectionally ranked (:func:`~alphaforge.ops.cs.cs_rank`).
            fwd_ret: Forward-return label panel (its rank is the target).
            horizon: Holding period used to build ``fwd_ret``.
            lag: Entry lag used to build ``fwd_ret``.

        Returns:
            Prediction panel on ``fwd_ret``'s index/columns: ``NaN`` up to and
            including the first refit date, and for cells where every feature
            is missing.  Predictions are in rank space (roughly ``(0, 1)``);
            they are not re-standardised.

        No look-ahead: a prediction at date ``d`` uses features of ``d`` and a
        model whose training labels were all realised on or before a refit
        date ``< d``.  Perturbing labels or features at dates ``>= d - lag -
        horizon`` cannot change predictions at dates ``<= d``.
        """
        if not isinstance(factors, Mapping) or len(factors) == 0:
            raise ValueError("factors must be a non-empty {name: panel} mapping")
        if not isinstance(fwd_ret, pd.DataFrame):
            raise TypeError("fwd_ret must be a pandas DataFrame panel")
        horizon = int(horizon)
        lag = int(lag)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")

        index, columns = fwd_ret.index, fwd_ret.columns
        n_dates, n_assets = fwd_ret.shape
        names = [str(n) for n in factors]
        self.feature_names_ = names
        self.refit_dates_ = []
        self.refit_history_ = []
        self.feature_importances_ = None
        self.last_model_ = None
        k = len(names)

        # ---- features: (T, N, K) tensor of cross-sectional ranks ---------
        feats = []
        for name, panel in factors.items():
            if not isinstance(panel, pd.DataFrame):
                raise TypeError(f"factor {name!r} must be a pandas DataFrame")
            if not (panel.index.equals(index) and panel.columns.equals(columns)):
                panel = panel.reindex(index=index, columns=columns)
            feats.append(cs_rank(panel).to_numpy(dtype=np.float64, na_value=np.nan))
        x_all = np.stack(feats, axis=-1) if feats else np.empty((n_dates, n_assets, 0))
        y_all = cs_rank(fwd_ret).to_numpy(dtype=np.float64, na_value=np.nan)

        any_feat = ~np.isnan(x_all).all(axis=-1) if k else np.zeros((n_dates, n_assets), dtype=bool)
        if self.model == "ridge":  # Ridge cannot take NaN: neutral mid-rank fill
            x_all = np.where(np.isnan(x_all), _RANK_FILL, x_all)
        usable = any_feat & np.isfinite(y_all)  # training-row mask
        date_known = usable.any(axis=1)  # date contributes >= 1 training row
        delay = lag + horizon

        pred = np.full((n_dates, n_assets), np.nan)
        cum_known = np.cumsum(date_known)
        enough = np.nonzero(cum_known >= self.min_train)[0]
        if enough.size == 0 or enough[0] + delay >= n_dates:
            logger.warning(
                "MLCombiner: never reached min_train=%d label-known dates (have %d, delay %d, %d dates); "
                "returning all-NaN predictions",
                self.min_train,
                int(cum_known[-1]) if n_dates else 0,
                delay,
                n_dates,
            )
            return pd.DataFrame(pred, index=index, columns=columns)
        first = int(enough[0]) + delay

        model: Any = None
        x_tr = y_tr = None
        for t in range(first, n_dates, self.refit_every):
            last_train = t - delay  # last position whose label is realised by t
            lo = max(0, last_train - self.train_window + 1)
            hi = last_train + 1
            m = usable[lo:hi].reshape(-1)
            x_tr = x_all[lo:hi].reshape(-1, k)[m]
            y_tr = y_all[lo:hi].reshape(-1)[m]
            if x_tr.shape[0] < 2:
                logger.warning("MLCombiner: refit at %s skipped - only %d training rows", index[t], x_tr.shape[0])
                continue
            tic = time.perf_counter()
            model = self._make_model()
            model.fit(x_tr, y_tr)
            fit_s = time.perf_counter() - tic

            p0, p1 = t + 1, min(t + self.refit_every + 1, n_dates)
            if p0 < p1:
                mp = any_feat[p0:p1].reshape(-1)
                x_p = x_all[p0:p1].reshape(-1, k)
                out = np.full(x_p.shape[0], np.nan)
                if mp.any():
                    out[mp] = model.predict(x_p[mp])
                pred[p0:p1] = out.reshape(p1 - p0, n_assets)

            self.refit_dates_.append(pd.Timestamp(index[t]))
            self.refit_history_.append(
                {
                    "date": pd.Timestamp(index[t]),
                    "n_rows": int(x_tr.shape[0]),
                    "train_start": pd.Timestamp(index[lo]),
                    "train_end": pd.Timestamp(index[last_train]),
                    "pred_start": pd.Timestamp(index[p0]) if p0 < n_dates else None,
                    "pred_end": pd.Timestamp(index[p1 - 1]) if p0 < p1 else None,
                    "fit_seconds": fit_s,
                }
            )
            logger.info(
                "MLCombiner[%s] refit at %s: n_rows=%d, n_features=%d, train %s..%s, predict %s..%s (%.2fs)",
                self.model,
                pd.Timestamp(index[t]).date(),
                x_tr.shape[0],
                k,
                pd.Timestamp(index[lo]).date(),
                pd.Timestamp(index[last_train]).date(),
                pd.Timestamp(index[p0]).date() if p0 < n_dates else "-",
                pd.Timestamp(index[p1 - 1]).date() if p0 < p1 else "-",
                fit_s,
            )

        if model is not None and x_tr is not None:
            self.last_model_ = model
            self.feature_importances_ = self._importances(model, x_tr, y_tr)
            logger.info(
                "MLCombiner: %d refits; importances %s",
                len(self.refit_dates_),
                {n: round(float(v), 4) for n, v in self.feature_importances_.items()},
            )
        return pd.DataFrame(pred, index=index, columns=columns)
