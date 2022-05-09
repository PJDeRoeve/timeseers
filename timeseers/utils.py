import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from pymc3 import Model, Normal, Slice, sample
from pymc3.distributions import Interpolated
from scipy.stats import halfcauchy


def dot(a, b):
    return (a * b[None, :]).sum(axis=-1)


class IdentityScaler:
    def fit(self, data):
        self.scale_factor_ = 1
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data):
        return data
    def inv_transform(self, series):
        return series



class MinMaxScaler:
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.max_ = data.max(axis=0)
            self.min_ = data.min(axis=0)
            self.scale_factor_ = (self.max_ - self.min_).where(self.max_ != self.min_, 1)
        if isinstance(data, np.ndarray):
            self.max_ = data.max(axis=0)[None, ...]
            self.min_ = data.min(axis=0)[None, ...]
            self.scale_factor_ = np.where(self.max_ != self.min_, self.max_ - self.min_, 1)
        if isinstance(data, pd.Series):
            self.max_ = data.max()
            self.min_ = data.min()
            self.scale_factor_ = self.max_ - self.min_

        return self

    def transform(self, series):
        return (series - self.min_) / self.scale_factor_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.scale_factor_ + self.min_


class MaxScaler:
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.scale_factor_ = data.max(axis=0)
        if isinstance(data, np.ndarray):
            self.scale_factor_ = data.max(axis=0)[None, ...]
        if isinstance(data, pd.Series):
            self.scale_factor_ = data.max()
        return self

    def transform(self, series):
        return series / self.scale_factor_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.scale_factor_


class StdScaler:
    def fit(self, data):
        if isinstance(data, pd.Series):
            self.mean_ = data.mean()
            self.std_ = data.std()

        return self

    def transform(self, series):
        return (series - self.mean_) / self.std_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.std_ + self.mean_


def add_subplot(height=5):
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)


def trend_data(n_changepoints, mu_k, sigma_k, mu_m, sigma_m, mu_delta, sigma_delta,
               location="spaced", noise=0.001,N=214, print_coefs=False):
    delta = np.random.laplace(loc=mu_delta, scale=sigma_delta, size=n_changepoints)

    t = np.linspace(0, 1, N)

    if location == "random":
        s = np.sort(np.random.choice(t, n_changepoints, replace=False))
    elif location == "spaced":
        s = np.linspace(0, np.max(t), n_changepoints + 2)[1:-1]
    else:
        raise ValueError('invalid `location`, should be "random" or "spaced"')

    A = (t[:, None] > s) * 1

    k, m = np.random.normal(mu_k, sigma_k), np.random.normal(mu_m, sigma_m)

    growth = k + A @ delta
    gamma = -s * delta
    offset = m + A @ gamma
    trend = growth * t + offset + np.random.randn(len(t)) * noise
    if print_coefs:
        print("k", k, "m", m, "delta", delta)

    return trend


def seasonal_data(n_components, mu_Beta, sigma_Beta, season_trend_balance,
                  noise=0.001, N=214, print_coefs=False):
    def X(t, p=52, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    t = np.linspace(0, 1, N)
    beta = np.random.normal(loc=mu_Beta, scale=sigma_Beta, size=2 * n_components)

    seasonality = X(t, 52 / len(t), n_components) @ beta + np.random.randn(len(t)) * noise
    seasonality = seasonality / season_trend_balance
    if print_coefs:
        print("beta", beta)
    return seasonality


def simulate_set(n_series, n_changepoints, n_components, noise_trend=0.001, noise_seasonality=0.001, mu_k=0, sigma_k=1,
                 mu_m=0, sigma_m=1, mu_delta=0, sigma_delta=0.2,
                 mu_Beta=0, sigma_Beta=1, season_trend_balance=1,N=214, print_coefs=False):
    values = np.empty(shape=(N * n_series))
    for i in range(0, n_series):
        simulated = trend_data(n_changepoints, noise=noise_trend, mu_k=mu_k, sigma_k=sigma_k, mu_m=mu_m,
                               sigma_m=sigma_m, mu_delta=mu_delta, sigma_delta=sigma_delta,print_coefs=print_coefs) + seasonal_data(n_components, noise=noise_seasonality, mu_Beta=mu_Beta,
                                                                sigma_Beta=sigma_Beta,
                                                                season_trend_balance=season_trend_balance,print_coefs=print_coefs)
        values[i * N:(i + 1) * N] = simulated

    organization_activity = np.repeat(a=list(range(0, n_series)), repeats=N)
    dates = np.tile(pd.date_range("2018-1-1", periods=N, freq="W"), reps=n_series)
    return pd.DataFrame({"t": dates, "value": values,
                         "organization_activity": organization_activity})


def logistic_growth_data(n_changepoints, location="spaced", noise=0.001, loc=0, scale=0.2):
    delta = np.random.laplace(size=n_changepoints, loc=loc, scale=scale)
    gamma = np.zeros(n_changepoints)

    t = np.linspace(0, 1, 1000)
    if location == "random":
        s = np.sort(np.random.choice(t, n_changepoints, replace=False))
    elif location == "spaced":
        s = np.linspace(0, np.max(t), n_changepoints + 2)[1:-1]
    else:
        raise ValueError('invalid `location`, should be "random" or "spaced"')

    A = (t[:, None] > s) * 1
    k, m = 2.5, 0

    for i in range(n_changepoints):
        left = (s[i] - m - np.sum(gamma[:i]))
        right = (1 - (k + np.sum(delta[:i])) / (k + np.sum(delta[:i+1])))
        gamma[i] = left * right

    g = (k + np.sum(A * delta, axis=1)) * (t - (m + np.sum(A * gamma, axis=1)))
    logistic_growth = 1 / (1 + np.exp(-g)) + np.random.randn(len(t)) * noise
    return (
        pd.DataFrame({"t": pd.date_range("2018-1-1", periods=len(t)), "value": logistic_growth}),
        delta,
    )



def rbf_seasonal_data(n_components, sigma=0.015, noise=0.001):
    def X(t, peaks, sigma, year):
        mod = (t % year)[:, None]
        left_difference = np.sqrt((mod - peaks[None, :]) ** 2)
        right_difference = np.abs(year - left_difference)
        return np.exp(- ((np.minimum(left_difference, right_difference)) ** 2) / (2 * sigma**2))

    t = pd.Series(pd.date_range("2010-01-01", "2014-01-01"))
    scaler = MinMaxScaler()
    scaled_t = scaler.fit_transform(t)
    scale_factor = t.max() - t.min()
    beta = np.random.normal(size=n_components)
    peaks = get_periodic_peaks(n_components)
    peaks = np.array([p / scale_factor for p in peaks])
    period = pd.Timedelta(days=365.25)
    seasonality = X(scaled_t, peaks, sigma, period / scale_factor) @ beta + np.random.randn(len(t)) * noise
    return (
        pd.DataFrame(
            {"t": pd.date_range("2018-1-1", periods=len(t)), "value": seasonality}
        ),
        beta,
    )


def get_group_definition(X, pool_cols, pool_type):
    if pool_type == 'complete':
        group = np.zeros(len(X), dtype='int')
        group_mapping = {0: 'all'}
    else:
        group = X[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(X[pool_cols].cat.categories))
    n_groups = X[pool_cols].nunique()
    return group, n_groups, group_mapping


def get_periodic_peaks(
        n: int = 20,
        period: pd.Timedelta = pd.Timedelta(days=365.25)):
    """
    Returns n periodic peaks that repeats each period. Return value
    can be used in RBFSeasonality.
    """
    return np.array([period * i / n for i in range(n)])

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))
