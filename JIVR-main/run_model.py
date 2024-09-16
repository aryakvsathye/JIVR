import pandas as pd
import numpy as np
from model import Implied_Volatility

betas = 'saved_model/implied_vol_model.csv'
conditional_var = 'saved_model/conditional_var.csv'
epsilon = 'saved_model/residuals.csv'
ts = 'saved_model/ts_model.json'
copula = 'saved_model/copula_corr.npy'
options_data = pd.read_csv('db.csv')
options_data['as_of'] = pd.to_datetime(options_data['as_of'])
options_data = options_data.loc[options_data['as_of'] >= '2022-12-28']
rates_data = options_data[['as_of', 'time_to_maturity', 'r']].drop_duplicates().sort_values(by=['as_of', 'time_to_maturity'])
spot_prices = options_data[['as_of', 'as_of_stock_price']].drop_duplicates().sort_values(by='as_of').set_index('as_of')
dividend_yield = options_data[['as_of', 'q']].drop_duplicates().sort_values(by='as_of').set_index('as_of')

implied_vol_model = Implied_Volatility(beta=betas,
                                       conditional_var=conditional_var,
                                       epsilon=epsilon,
                                       ts=ts,
                                       copula=copula,
                                       S=spot_prices,
                                       r=rates_data,
                                       q=dividend_yield)
implied_vol_model.forecast(pd.Timestamp(year=2023, month=4, day=28))
implied_vol_model.beta.to_csv('results/predicted_beta.csv')
implied_vol_model.h.to_csv('results/predicted_h.csv')
# print(implied_vol_model.predict_volatility(20, 400, pd.Timestamp(year=2023, month=4, day=20)))