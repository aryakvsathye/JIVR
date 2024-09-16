import pandas as pd
from scipy.stats import norm
import numpy as np
from model.implied_vol import Implied_Volatility


def get_actual_vix(t, tau, filename):
    """
    Extracts the exact VIX value from known VIX data
    """
    df = pd.read_csv(filename)
    vix = df.loc[t]["Adj Close"]

    return vix


def find_nearest_tau(df, tau=30):
    tau_arr = df['time_to_maturity'].to_numpy()
    sorted_tau_arr = np.sort(tau_arr)
    sorted_bln_tau_arr = sorted_tau_arr <= tau
    # index location of tau2 is the first occurrence of a time to maturity greater than tau
    tau2_idx = np.where(sorted_bln_tau_arr == 0)[0][0]
    tau2 = sorted_tau_arr[tau2_idx]
    tau1 = sorted_tau_arr[tau2_idx - 1]

    return tau1, tau2


def get_call_put(df):
    full_ticker = df['ticker'].str.split().str[-1].to_numpy().astype('str')
    # array with value 1 if call option, 0 otherwise
    call_put_bln = np.char.startswith(full_ticker, 'C')
    return np.where(call_put_bln == 1, 'c', 'p')


def calculate_black_scholes_price(call_put, S, r, q, tau, K, sigma):
    """
    Calculates Black-Scholes option price with dividends according to
    equations 17.4, 17.5 in Hull
    """
    N = norm.cdf
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if call_put == 'c':
        return S * np.exp(-q * tau) * N(d1) - K * np.exp(-r * tau) * N(d2)

    if call_put == 'p':
        return K * np.exp(-r * tau) * N(-d2) - S * np.exp(-q * tau) * N(-d1)


def calculate_option_price(call_put, S, r, q, tau, K):
    """
    Calculates price of option by Black Scholes formula, using the implied volatility
    generated by the JIVR model
    :param S: spot price
    :param r: interest rate
    :param q: dividend rate
    :param tau: time to maturity
    :param K: strike price
    :param t: time in days
    :return: price of option
    """
    implied_vol_model = Implied_Volatility()
    sigma = implied_vol_model.predict(S, r, q, tau, K)
    black_scholes_price = calculate_black_scholes_price(call_put, S, r, q, tau, K, sigma)

    return black_scholes_price
