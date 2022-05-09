import numpy as np
import pandas as pd
import pymc3 as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition


class FourierSeasonality(TimeSeriesModel):
    def __init__(
        self,
        name: str = None,
        n: int = 10,
        period: pd.Timedelta = pd.Timedelta(days=365.25),
        shrinkage_strength=10,
        pool_cols=None,
        pool_type='none'
    ):
        self.n = n
        self.period = period
        self.shrinkage_strength = shrinkage_strength
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"FourierSeasonality(period={self.period})"
        super().__init__()

    @staticmethod
    def _X_t(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        group, self.n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)

        self.p_ = self.period / scale_factor['t']
        n_params = self.n * 2

        with model:
            if self.pool_type == 'partial':
                mu_beta = pm.Normal(self._param_name("mu_beta"), mu=0, sigma=1, shape=n_params)
                sigma_beta = pm.HalfNormal(self._param_name("sigma_beta"), 0.1, shape=n_params)
                offset_beta = pm.Normal(
                    self._param_name("offset_beta"),
                    0,
                    1 / self.shrinkage_strength,
                    shape=(len(self.groups_), n_params)
                )

                beta = pm.Deterministic(self._param_name("beta"), mu_beta + offset_beta * sigma_beta)
            else:
                beta = pm.Normal(self._param_name("beta"), 0, 1, shape=(len(self.groups_), n_params))

            seasonality = pm.math.sum(self._X_t(t, self.p_, self.n) * beta[group], axis=1)

        return seasonality

    def _predict(self, trace, t, pool_group=None):
        if pool_group == None:
            group = 0
        else:
            group = pool_group
        return self._X_t(t, self.p_, self.n) @ trace[self._param_name("beta")][:, group].T

    def predict(self, trace, scaled_t, y_scaler, pool_group=None, multi=False):
        seasonality_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            scaled_s = self._predict(trace, scaled_t, group_code)
            seasonality_return[:, group_code] = scaled_s.mean(axis=1)
        # If a specific pool_group was given, we only subset on those predictions
        if self.pool_type == 'complete':
            seasonality_return = seasonality_return.reshape(-1)
            if pool_group == None:
                seasonality_return = np.transpose(np.tile(seasonality_return, ((self.n_groups),1)))
        else:
            if pool_group != None:
                seasonality_return = seasonality_return[:, pool_group]
        return seasonality_return

    def plot(self, trace, scaled_t, y_scaler, ax,idx_tracker, groups_subset, multi=False):
        ax[idx_tracker["idx"]].set_title(str(self))

        groups_plot = {}
        if self.pool_type != 'complete':
            if groups_subset is not None:
                for group_code, group_name in self.groups_.items():
                    if group_name in groups_subset:
                        groups_plot[group_name] = group_code
            else:
                groups_plot = self.groups_.copy()
                groups_plot = {v: k for k, v in groups_plot.items()}
        else:
            for group in y_scaler.keys():
                if groups_subset is None or group in groups_subset:
                    groups_plot[group] = 0
        seasonality_return = np.empty((len(scaled_t), len(y_scaler.keys())))
        for group_name, group_code in groups_plot.items():
            scaled_s = self._predict(trace, scaled_t, group_code)
            if multi:
                s = 1 + scaled_s
            else:
                s = y_scaler[group_name].inv_transform(scaled_s)
            ax[idx_tracker["idx"]].plot(list(range(self.period.days)), s.mean(axis=1)[:self.period.days], label=group_name)
            seasonality_return[:, group_code] = scaled_s.mean(axis=1)


        idx_tracker["idx"] += 1

        return seasonality_return

    def __repr__(self):
        return f"FourierSeasonality(n={self.n}, " \
               f"period={self.period}," \
               f"pool_cols={self.pool_cols}, " \
               f"pool_type={self.pool_type}"
