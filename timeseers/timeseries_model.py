import pandas as pd
import pymc3 as pm
from timeseers.utils import MinMaxScaler, StdScaler, hour_rounder, IdentityScaler
from timeseers.likelihood import Gaussian
import numpy as np
from abc import ABC, abstractmethod
import re


class TimeSeriesModel(ABC):
    def fit(self, X, y, X_scaler=MinMaxScaler, scaler=StdScaler, likelihood=None,MAP=False, **sample_kwargs):
        if not X.index.is_monotonic_increasing:
            raise ValueError('index of X is not monotonically increasing. You might want to call `.reset_index()`')
        X_to_scale = X.select_dtypes(exclude='category')
        X_groups = X.select_dtypes('category')
        self._X_scaler_ = X_scaler()
        X_scaled = self._X_scaler_.fit_transform(X_to_scale)
        #EXTRA APPARTE SCALERS
        self._y_scaler_list = {}
        y_scaled = y.copy()
        for group in np.unique(X_groups.values):
            self._y_scaler_list[group] = scaler()
            index_group = X_groups[X_groups.values == group].index
            y_scaled.loc[index_group] =  self._y_scaler_list[group].fit_transform(y.loc[index_group])

        model = pm.Model()
        X_scaled = X_scaled.join(X.select_dtypes('category'))
        del X
        mu = self.definition(
            model, X_scaled, self._X_scaler_.scale_factor_
        )

        if likelihood is None:
            likelihood = Gaussian()
        with model:
            likelihood.observed(mu, y_scaled)
            if MAP:
                self.trace_ = pm.find_MAP(**sample_kwargs)
                for var in self.trace_.keys():
                    self.trace_[var] = np.expand_dims(self.trace_[var], axis=0)
            else:
                self.trace_ = pm.sample(**sample_kwargs)
            self.prior_predictive = pm.sample_prior_predictive(samples=500, random_seed=62)
            return self.trace_


#To be done -> auto determine the number of components
    def plot_components(self, X_true=None, y_true=None, groups=None, y_scaled=None,fig=None, freq_set='D', num_component=None, groups_subset=None):
        plot_indicator = {"idx":0}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(num_component, 1, figsize=(8,24))
        lookahead_scale = 0.3
        t_min, t_max = self._X_scaler_.min_["t"], self._X_scaler_.max_["t"]
        t_max += (t_max - t_min) * lookahead_scale
        t = pd.date_range(t_min, t_max, freq=freq_set)
        scaled_t = np.linspace(0, 1 + lookahead_scale, len(t))
        #Plot components
        total = self.plot(self.trace_, scaled_t, self._y_scaler_list, ax=ax, idx_tracker=plot_indicator, groups_subset=groups_subset)
        ax[plot_indicator["idx"]].set_title("overall")
        ax[plot_indicator["idx"]].plot(t, total)
        if X_true is not None and y_true is not None:
            if groups is not None:
                group_loop = groups.cat.categories
                if groups_subset is not None:
                    group_loop = groups_subset
                for group in group_loop:
                    mask = groups == group
                    ax[plot_indicator["idx"]].scatter(X_true["t"][mask], y_true[mask], label=group, marker='.',
                                                      alpha=0.2)
            else:
                ax[plot_indicator["idx"]].scatter(X_true["t"], y_scaled, c="k", marker='.', alpha=0.2)
        plt.show()

    #PREDICTION METHOD

    #TO-DO: REGARDLESS OF POOL TYPE
    def predict_timesteps(self, step_length="7D", pool_group=None):
        #Extracting the time unit by which we wish to predict
        FREQ_UNIT = "".join(re.split("[^a-zA-Z]*", step_length))
        #Getting the first and last timestamps from the dataset on which the model was fitted
        t_min, t_max = self._X_scaler_.min_["t"], self._X_scaler_.max_["t"]
        #During training the timestamps where scaled between 0-1.
        #The lookahead_scale is the numerical translation of the timestamps for which we which to predict
        lookahead_scale = pd.Timedelta(step_length) / (t_max - t_min)
        #Correction factor so we start to predict 1 timeunit after the last training timestamp
        lookahead_correction = pd.Timedelta(f"1{FREQ_UNIT}") / (t_max - t_min)
        #t_plus is the (last) timestamp for which we wish to predict. t_max - t_min should always be = 1
        t_plus = t_max + (t_max - t_min) * lookahead_scale
        #Rounding t_plus as it sometimes there is a very small rounding error
        t_plus = hour_rounder(t_plus)
        #Getting the dates for which we wish to make predictions
        t = pd.date_range(t_max + pd.Timedelta(f"1{FREQ_UNIT}"), t_plus, freq=FREQ_UNIT)
        #Getting the scaled numeric timestamps
        scaled_t = np.linspace(1 + lookahead_correction, 1 + lookahead_scale + lookahead_correction, len(t) + 1)
        preds = self.predict(self.trace_, scaled_t, self._y_scaler_list, pool_group=pool_group)
        return preds[1:14,:]



    @abstractmethod
    def plot(self, trace, t, y_scaler, ax, idx_tracker, groups_plot):
        pass

    @abstractmethod
    def predict(self, trace, t, y_scaler, pool_group):
        pass

    @abstractmethod
    def definition(self, model, X_scaled, scale_factor):
        pass

    def _param_name(self, param):
        return f"{self.name}-{param}"

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __str__(self):
        return self.name


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) + self.right.definition(
            *args, **kwargs
        )
    def plot(self, trace, scaled_t, y_scaler, ax, idx_tracker, groups_subset, multi=False):
        left = self.left.plot(trace, scaled_t, y_scaler, ax,idx_tracker,groups_subset, multi=False)
        right = self.right.plot(trace, scaled_t, y_scaler, ax,idx_tracker,groups_subset, multi=False)
        group_dict = self.left.groups_
        group_dict_right =  self.right.groups_
        if len(group_dict_right.keys()) > len(group_dict.keys()):
            group_dict = group_dict_right
        preds = left + right
        idx_list = []
        if group_dict[0] != "all":
            for idx,group in group_dict.items():
                #Group code
                if (groups_subset is None) or (group in groups_subset) :
                    idx_list.append(idx)
                    preds[:,idx] = y_scaler[group].inv_transform(preds[:,idx])
        else:
            for idx,group in enumerate(y_scaler.keys()):
                if (groups_subset is None) or (group in groups_subset) :
                    idx_list.append(idx)
                    preds[:,idx] = y_scaler[group].inv_transform(preds[:,idx])
        return preds[:,idx_list]


    # NOG AANPASSEN
    def predict(self, trace, scaled_t, y_scaler, pool_group, multi=False):
        left = self.left.predict(trace, scaled_t, y_scaler, multi=False)
        right = self.right.predict(trace, scaled_t, y_scaler,  multi=False)
        group_dict = self.left.groups_
        group_dict_right =  self.right.groups_
        if len(group_dict_right.keys()) > len(group_dict.keys()):
            group_dict = group_dict_right
        if group_dict[0] == "all":
            group_dict = {}
            for i in range(0, len(y_scaler.keys())):
                group_dict[i] = list(y_scaler.keys())[i]
        idx = 0
        preds = left + right
        for idx,group in group_dict.items():
            #Group code
            preds[:,idx] = y_scaler[group].inv_transform(preds[:,idx])
        return preds

    def __repr__(self):
        return (
            f"AdditiveTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) * (
            1 + self.right.definition(*args, **kwargs)
        )

    def plot(self, trace, scaled_t, y_scaler, ax, idx_tracker, groups_subset, multi=True):
        left = self.left.plot(trace, scaled_t, y_scaler, ax,idx_tracker,groups_subset, multi=False)
        right = self.right.plot(trace, scaled_t, y_scaler, ax,idx_tracker,groups_subset, multi=False)
        group_dict = self.left.groups_
        group_dict_right =  self.right.groups_
        if len(group_dict_right.keys()) > len(group_dict.keys()):
            group_dict = group_dict_right
        preds = left + left*right
        for idx,group in group_dict.items():
            #Group code
            if (groups_subset is None) or (group in groups_subset) :
                preds[:,idx] = y_scaler[group].inv_transform(preds[:,idx])
        return preds

    def predict(self, trace, scaled_t, y_scaler, pool_group, multi=True):
        left = self.left.predict(trace, scaled_t, y_scaler, multi=False)
        right = self.right.predict(trace, scaled_t, y_scaler,  multi=False)
        group_dict = self.left.groups_
        group_dict_right =  self.right.groups_
        if len(group_dict_right.keys()) > len(group_dict.keys()):
            group_dict = group_dict_right
        inv_map = {v: k for k, v in group_dict.items()}
        idx = 0
        preds = left + left*right
        for idx,group in group_dict.items():
            #Group code
            preds[:,idx] = y_scaler[group].inv_transform(preds[:,idx])
        return preds

    def __repr__(self):
        return (
            f"MultiplicativeTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )

