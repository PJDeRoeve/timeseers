from abc import ABC, abstractmethod
import pymc3 as pm


class Likelihood(ABC):
    """Subclasses should implement the observed method which defines an observed random variable"""
    @abstractmethod
    def observed(self, mu, y_scaled):
        pass


class Gaussian(Likelihood):
    """Gaussian likelihood with constant variance"""
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def observed(self, mu, y_scaled):
        sigma = pm.HalfCauchy("sigma", self.sigma)
        pm.Normal("obs", mu=mu, sd=sigma, observed=y_scaled)
class AR1LH(Likelihood):
    """AR1 Likelyhood with tau as the precision"""
    def __init__(self, tau=0.5, center_mu=0, center_sigma=1):
        self.tau = tau
        self.center_mu = center_mu
        self.center_sigma = center_sigma
    def observed(self, y_scaled):
        tau = pm.Exponential("tau", self.tau)
        center = pm.Normal("center", mu=self.center_mu, sigma=self.center_sigma)
        theta = pm.Normal("theta", 0.0, 1.0)
        pm.AR1("obs", k=theta, tau_e=tau, observed= y_scaled - center)





class StudentT(Likelihood):
    """StudentT likelihood with constant variance, robust to outliers"""
    def __init__(self, alpha=1., beta=1., sigma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def observed(self, mu, y_scaled):
        nu = pm.InverseGamma("nu", alpha=self.alpha, beta=self.beta)
        sigma = pm.HalfCauchy("sigma", self.sigma)
        pm.StudentT("obs", mu=mu, sd=sigma, nu=nu, observed=y_scaled)
