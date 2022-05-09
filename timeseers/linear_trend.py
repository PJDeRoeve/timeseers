import numpy as np
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition
import pymc3 as pm


class LinearTrend(TimeSeriesModel):
    def __init__(
            self, name: str = None, n_changepoints=None, changepoints_prior_scale=0.05, growth_prior_scale=1,
            pool_cols=None, pool_type='none'
    ):
        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.growth_prior_scale = growth_prior_scale
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"LinearTrend(n_changepoints={n_changepoints})"
        super().__init__()

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        group, self.n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)
        self.s = np.linspace(0, np.max(t), self.n_changepoints + 2)[1:-1]

        with model:
            A = (t[:, None] > self.s) * 1.0

            if self.pool_type == 'partial':
                mu_k = pm.Normal(self._param_name("mu_k"), mu=0, sigma=5)
                sigma_k = pm.HalfCauchy(self._param_name('sigma_k'), beta=self.growth_prior_scale)
                offset_k = pm.Normal(self._param_name('offset_k'), mu=0, sd=1, shape=len(self.groups_))
                k = pm.Deterministic(self._param_name("k"), mu_k + offset_k * sigma_k)

                mu_delta = pm.Normal(self._param_name("mu_delta"), mu=0, sigma=5)
                sigma_delta = pm.HalfCauchy(self._param_name('sigma_delta'), beta=self.changepoints_prior_scale)
                offset_delta = pm.Laplace(self._param_name('offset_delta'), 0, 1,
                                          shape=(len(self.groups_), self.n_changepoints))
                delta = pm.Deterministic(self._param_name("delta"), mu_delta + offset_delta * sigma_delta)

            else:
                delta = pm.Laplace(
                    self._param_name("delta"), 0, self.changepoints_prior_scale,
                    shape=(len(self.groups_), self.n_changepoints)
                )
                k = pm.Normal(self._param_name("k"), 0, self.growth_prior_scale, shape=len(self.groups_))

            m = pm.Normal(self._param_name("m"), 0, 5, shape=len(self.groups_))

            gamma = -self.s * delta[group, :]

            g = (
                    (k[group] + pm.math.sum(A * delta[group], axis=1)) * t
                    + (m[group] + pm.math.sum(A * gamma, axis=1))
            )
        return g
    def _predict(self, trace, t, pool_group=None):
        A = (t[:, None] > self.s) * 1
        if pool_group == None:
            group = 0
        else:
            group = pool_group
        k, m = trace[self._param_name("k")][:, group], trace[self._param_name("m")][:, group]
        growth = k + A @ trace[self._param_name("delta")][:, group].T
        gamma = -self.s[:, None] * trace[self._param_name("delta")][:, group].T
        offset = m + A @ gamma
        return growth * t[:, None] + offset

    def plot(self, trace, scaled_t, y_scaler_list, ax, idx_tracker, groups_subset, multi=False):
        ax[idx_tracker["idx"]].set_title(str(self))
        ax[idx_tracker["idx"]].set_xticks([])
        # Getting all the groups we want to plot in a dictionary called groups plot
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
            for group in y_scaler_list.keys():
                if groups_subset is None or group in groups_subset:
                    groups_plot[group] = 0
        trend_return = np.empty((len(scaled_t), len(y_scaler_list.keys())))
        for group_name, group_code in groups_plot.items():
            scaled_trend = self._predict(trace, scaled_t, group_code)
            trend = y_scaler_list[group_name].inv_transform(scaled_trend)
            ax[idx_tracker["idx"]].plot(scaled_t, trend.mean(axis=1), label=group_name)
            trend_return[:, group_code] = scaled_trend.mean(axis=1)
        for changepoint in self.s:
            ax[idx_tracker["idx"]].axvline(changepoint, linestyle="--", alpha=0.2, c="k")
        ax[idx_tracker["idx"]].legend()
        idx_tracker["idx"] += 1

        return trend_return
    #TO-DO: Not so efficient as predictions for all groups always get calculated.
    def predict(self, trace, scaled_t, y_scaler_list, pool_group=None, multi=False):
        trend_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            scaled_trend = self._predict(trace, scaled_t, group_code)
            trend_return[:, group_code] = scaled_trend.mean(axis=1)
        #If a specific pool_group was given, we only subset on those predictions
        if self.pool_type == 'complete':
            trend_return = trend_return.reshape(-1)
            if pool_group == None:
                trend_return = np.transpose(np.tile(trend_return, ((self.n_groups),1)))
        else:
            if pool_group != None:
                trend_return = trend_return[:, pool_group]
        return trend_return
    def __repr__(self):
        return f"LinearTrend(n_changepoints={self.n_changepoints}, " \
               f"changepoints_prior_scale={self.changepoints_prior_scale}, " \
               f"growth_prior_scale={self.growth_prior_scale})"
