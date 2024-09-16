import numpy as np
from scipy.stats import norminvgauss

def forward_price(S, r_t_tau, q_t, tau):
    return np.multiply(S, np.exp(np.multiply(r_t_tau - q_t, tau)))

def moneyness(tau, F_t_tau, K):
    return np.multiply(np.power(tau, -0.5), np.log(np.divide(F_t_tau, K)))

def time_to_maturity(tau, T_conv):
    return np.exp(-np.power(np.divide(tau, T_conv), 1/2))

def moneyness_slope(M):
    return np.where(M >= 0, M, np.divide(np.exp(2*M) - 1, np.exp(2*M) + 1))

def smile_attenuation(M, tau, T_max):
    return np.multiply(1 - np.exp(-np.power(M, 2)), np.log(np.divide(tau, T_max)))

def smirk(M, tau, T_max):
    return np.where(M < 0, np.multiply(1 - np.exp(np.power(3*M, 3)), np.log(np.divide(tau, T_max))), 0)

def nig_convert_params(zeta, phi):
    alpha = np.sqrt(np.power(phi, 2) + np.power(zeta, 2))
    mu = -np.divide(np.multiply(np.power(phi, 2), zeta), np.power(phi, 2) + np.power(zeta, 2))
    delta = np.divide(np.power(phi, 3), np.power(phi, 2) + np.power(zeta, 2))
    beta = zeta
    gamma = phi
    a = alpha*delta
    b = beta*delta
    return alpha, beta, mu, delta

def nig_innovation(q, zeta, phi):
    alpha, beta, mu, delta = nig_convert_params(zeta, phi)
    return norminvgauss.ppf(q, alpha*delta, beta*delta, loc=mu, scale=delta)

def get_closest_tau(tau, times):
    times_ = np.abs(times - tau)
    return times[np.argmin(times_)]
