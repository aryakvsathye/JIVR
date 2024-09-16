import numpy as np
import pandas as pd

from helper_functions import calculate_option_price, get_call_put, find_nearest_tau, get_actual_vix
from model.functions import forward_price


def vix(t, tau, data):
    """
    Returns the VIX index from a portfolio of put and call options for a given
    time to maturity tau, according to equation (6) in section 6.2
    :param t: time
    :param tau: time-to-maturity
    :param data: pandas dataframe of options

    :return: VIX index price
    """
    # extract only the relevant options with matching time to maturity
    options_df = data[data["time_to_maturity"] == tau]
    options_df_sorted = options_df.sort_values(by=["strike"])
    N = len(options_df_sorted.index)  # number of options
    # K = options_df_sorted["strike"].to_numpy()
    K = np.linspace(300, 500, 5, dtype=int)
    r = options_df_sorted["r"].to_numpy()
    S = options_df_sorted["as_of_stock_price"].to_numpy()
    q = options_df_sorted["q"].to_numpy()
    # to get whether each option is a call 'c' or put 'p'
    call_put = get_call_put(options_df_sorted)
    # calculate delta strike
    delta_K = (K[2:] - K[:-2]) / 2
    # add initial and final endpoints of array
    delta_K = np.insert(delta_K, 0, K[1] - K[0])
    delta_K = np.append(delta_K, K[-1] - K[-2])

    inner_sum = 0
    for i in range(N):
        option_price = calculate_option_price(pd.Series(call_put[i]), pd.Series(S[i]), pd.Series(r[i]), pd.Series(q[i]),
                                              pd.Series(tau), pd.Series(K[i]))

        # NOTE: for the below, we set K_(j, tau) in eqn 6 equal to the forward price, so
        # the second term is equal to 0
        inner_sum += (delta_K[i] / (K[i] ** 2)) * np.exp(r[i] * tau) * option_price
        # inner_sum += (delta_K[i] / (K[i] ** 2)) * np.exp(r[i] * tau) * option_price - \
        #              (1 / tau) * ((forward_price(S[i], r[i], q[i], tau) / K[i]) - 1) ** 2

    return 100 * np.sqrt((2 / tau) * inner_sum)


# def vix_index_prediction(t, tau, data):
#     """
#     Calculates the VIX index prediction for T=30 days by linear interpolation
#     between VIX predictions between two nearest time-to-maturities surrounding tau
#     """
#     tau1, tau2 = find_nearest_tau(data, tau)
#
#     term1 = ((tau2 - tau) / (tau2 - tau1)) * (vix(t, tau1, data)) ** 2
#     term2 = ((tau - tau1) / (tau2 - tau1)) * (vix(t, tau2, data)) ** 2
#
#     return np.sqrt(term1 + term2)
def vix_forecast_model(t, d, tau, data):
    """
    Generates forecasts for the VIX variation over a prediction horizon of d days
    :param t: particular date
    :param d: number of business days
    :param tau: time to maturity
    :return: float, the VIX forecast
    """
    return vix(t + d, tau, data) - vix(t, tau, data)


def vix_index_forecast(t, d, tau, data, vix_data):
    """
    Generates the actual VIX Index forecast using both the JIVR model and actual VIX data
    """
    vix_actual = get_actual_vix(t, d, tau, vix_data)

    return vix_actual + vix_forecast_model(t, d, tau, data)
