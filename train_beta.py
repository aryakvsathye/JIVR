import pandas as pd
import numpy as np
from model import Implied_Volatility

data = pd.read_csv('db.csv')
data['exp_date'] = pd.to_datetime(data['exp_date'])
data['as_of'] = pd.to_datetime(data['as_of'])
data = data.loc[data['as_of'] >= '2022-12-28']
data = data.set_index('as_of')
print("Dataset loaded, number of rows =", len(data))

bakshi_et_al_exclusions = True

if bakshi_et_al_exclusions:
    data = data.loc[~(data['time_to_maturity'] < 6)]
    print("Number of options left after removing near maturity options =", len(data))
    data = data.loc[~(data['bid'] == 0)]
    print("Number of options left after removing bid 0 =", len(data))
    data = data.loc[~(data['ask'] - data['bid'] > 1.75*(data['ask'] + data['bid'])/2)]
    print("Number of options left after removing excess bid-ask spread =", len(data))

data['time_to_maturity'] = data['time_to_maturity']/252
data['r'] = data['r']*252
data['q'] = data['q']*252
implied_vol_model = Implied_Volatility()
implied_vol_model.fit(data['ivm']/100, data['as_of_stock_price'], data['r'], data['q'], data['time_to_maturity'], data['strike'])
print(implied_vol_model.beta)

stats = pd.DataFrame(data={'Min':[], 'Q1':[], 'Median':[], 'Q3':[], 'Max':[], 'Mean':[], 'Std':[], 'Skew':[], 'Kurt':[]})
for i in range(5):
    min = np.min(implied_vol_model.beta.iloc[:, i])
    q1 = np.quantile(implied_vol_model.beta.iloc[:, i], 0.25)
    median = np.quantile(implied_vol_model.beta.iloc[:, i], 0.5)
    q3 = np.quantile(implied_vol_model.beta.iloc[:, i], 0.75)
    max = np.max(implied_vol_model.beta.iloc[:, i])
    mn = np.mean(implied_vol_model.beta.iloc[:, i])
    stddev = np.std(implied_vol_model.beta.iloc[:, i])
    skew = implied_vol_model.beta.iloc[:, i].skew()
    kurt = implied_vol_model.beta.iloc[:, i].kurtosis()
    stats = pd.concat([stats, pd.DataFrame(data={'Min':[min], 'Q1':[q1], 'Median':[median], 'Q3':[q3], 'Max':[max], 'Mean':[mn], 'Std':[stddev], 'Skew':[skew], 'Kurt':[kurt]}, index=['beta'+str(i+1)])])

print(stats)
implied_vol_model.save('saved_model/implied_vol_model.csv')