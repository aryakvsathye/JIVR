from .functions import forward_price, moneyness, time_to_maturity, moneyness_slope, smile_attenuation, smirk, nig_innovation, get_closest_tau
from copulae import GaussianCopula
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BusinessDay
import json

class Implied_Volatility(object):

    def __init__(self, beta=None, conditional_var=None, epsilon=None, ts=None, copula=None, S=None, r=None, q=None, T_conv=0.25, T_max=5):
        if type(beta) is str:
            self.beta = pd.read_csv(beta)
            self.beta['as_of'] = pd.to_datetime(self.beta['as_of'])
            self.beta = self.beta.set_index('as_of')
            self.beta_std = {'beta_2':np.std(self.beta.iloc[:, 1]), 'beta_5':np.std(self.beta.iloc[:, 4])}
        else:
            self.beta = beta
        if type(conditional_var) is str:
            self.h = pd.read_csv(conditional_var)
            self.h = self.h.set_index(self.beta.index[1:])
        else:
            self.h = conditional_var
        if type(epsilon) is str:
            self.e = pd.read_csv(epsilon)
            self.e = self.e.set_index(self.beta.index[1:])
        else:
            self.e = epsilon
        if type(ts) is str:
            f = open(ts)
            self.ts = json.load(f)
        else:
            self.ts = ts
        if type(copula) is str:
            self.copula = GaussianCopula(dim=6)
            self.copula[:] = np.load(copula)
        else:
            self.copula = copula
        self.S = S
        self.r = r
        self.q = q
        self.T_conv = T_conv
        self.T_max = T_max

    def predict_volatility(self, tau, K, t):
        if t not in self.beta.index:
            self.forecast(t)
        S_t = self.S.loc[t].to_numpy()[0]
        q_t = self.q.loc[t].to_numpy()[0]
        tau_r = get_closest_tau(tau, self.r.loc[self.r['as_of'] == t, 'time_to_maturity'].to_numpy())
        r_t_tau = self.r.loc[(self.r['as_of'] == t) & (self.r['time_to_maturity'] == tau_r), 'r'].to_numpy()[0]
        F_t_tau = forward_price(S_t, r_t_tau, q_t, tau)
        M = moneyness(tau, F_t_tau, K)
        sigma_t = self.beta.loc[t, 'beta_1'] + \
                np.multiply(self.beta.loc[t, 'beta_2'], time_to_maturity(tau, self.T_conv)) + \
                np.multiply(self.beta.loc[t, 'beta_3'], moneyness_slope(M)) + \
                np.multiply(self.beta.loc[t, 'beta_4'], smile_attenuation(M, tau, self.T_max)) + \
                np.multiply(self.beta.loc[t, 'beta_5'], smirk(M, tau, self.T_max))
        return sigma_t
    
    def get_spot_price(self, t):
        if t not in self.beta.index:
            self.forecast(t)
        return self.S.loc[t].to_numpy()[0]

    def get_risk_free_rate(self, t, tau):
        if t not in self.beta.index:
            self.forecast(t)
        tau_r = get_closest_tau(tau, self.r.loc[self.r['as_of'] == t, 'time_to_maturity'])
        return self.r.loc[(self.r['as_of'] == t) & (self.r['time_to_maturity'] == tau_r), 'r'].to_numpy()[0]

    def get_dividend_yield(self, t):
        if t not in self.beta.index:
            self.forecast(t)
        return self.q.loc[t].to_numpy()[0]

    def forecast(self, t):
        if t-BusinessDay(n=1) not in self.beta.index:
            self.forecast(t-BusinessDay(n=1))
        S_t_1, beta_t_1, h_t_1, e_t_1, q_t, r_t_tau = self._forecast_for_t(t)
        S_t_1 = pd.DataFrame(data={'as_of_stock_price':[S_t_1[0]]}, index=[t])
        self.S = self.S.append(S_t_1)
        beta_t_1 = pd.DataFrame(data={'beta_1':[beta_t_1[0]], 'beta_2':[beta_t_1[1]], 'beta_3':[beta_t_1[2]], 'beta_4':[beta_t_1[3]], 'beta_5':[beta_t_1[4]]}, index=[t])
        self.beta = self.beta.append(beta_t_1)
        h_t_1 = pd.DataFrame(data={'h_y':[h_t_1[0]], 'h_beta_1':[h_t_1[1]], 'h_beta_2':[h_t_1[2]], 'h_beta_3':[h_t_1[3]], 'h_beta_4':[h_t_1[4]], 'h_beta_5':[h_t_1[5]]}, index=[t])
        self.h = self.h.append(h_t_1)
        e_t_1 = pd.DataFrame(data={'y_residuals':[e_t_1[0]], 'beta_1_residuals':[e_t_1[1]], 'beta_2_residuals':[e_t_1[2]], 'beta_3_residuals':[e_t_1[3]], 'beta_4_residuals':[e_t_1[4]], 'beta_5_residuals':[e_t_1[5]]}, index=[t])
        self.e = self.e.append(e_t_1)
        self.q = self.q.append(pd.DataFrame(data={'q':[q_t[0]]}, index=[t]))
        self.r = self.r.append(pd.DataFrame(data={'as_of':[t], 'time_to_maturity':[252], 'r':[r_t_tau[0]]}))
    
    def _forecast_for_t(self, t):
        S_t = self.S.loc[t-BusinessDay(n=1)].to_numpy()
        q_t = self.q.loc[t-BusinessDay(n=1)].to_numpy()
        h_t = self.h.loc[t-BusinessDay(n=1), :]
        e_t = self.e.loc[t-BusinessDay(n=1), :]
        beta_t = self.beta.loc[t-BusinessDay(n=1), :]
        beta_t2 = self.beta.loc[t-BusinessDay(n=2), :]
        tau = 252
        tau_r = get_closest_tau(tau, self.r.loc[self.r['as_of'] == t-BusinessDay(n=1), 'time_to_maturity'].to_numpy())
        r_t_tau = self.r.loc[(self.r['as_of'] == t-BusinessDay(n=1)) & (self.r['time_to_maturity'] == tau_r), 'r'].to_numpy()
        
        innov_cdf = self.copula.random(1)
        ATM_1mo_IV_t = self.beta.loc[t-BusinessDay(n=1), 'beta_1'] + self.beta.loc[t-BusinessDay(n=1), 'beta_2']*time_to_maturity(21, self.T_conv)

        epsilon_beta_1 = nig_innovation(innov_cdf[1], self.ts['beta_1']['skew'][0], self.ts['beta_1']['shape'][0])
        h_beta_1_t_1 =  self.ts['beta_1']['alpha1'][0]*h_t['h_beta_1']*np.power(e_t['beta_1_residuals'] - self.ts['beta_1']['eta21'][0], 2)
        beta_1_t_1 = self.ts['beta_1']['ar1'][0]*beta_t['beta_1'] + \
                        self.ts['beta_1']['mxreg1'][0]*beta_t['beta_2'] + \
                        np.sqrt(h_beta_1_t_1)*epsilon_beta_1
        
        epsilon_beta_2 = nig_innovation(innov_cdf[2], self.ts['beta_2']['skew'][0], self.ts['beta_2']['shape'][0])
        h_beta_2_t_1 =  self.ts['beta_2']['alpha1'][0]*h_t['h_beta_2']*np.power(e_t['beta_2_residuals'] - self.ts['beta_2']['eta21'][0], 2)
        beta_2_t_1 = self.ts['beta_2']['mu'][0] + \
                        self.ts['beta_2']['ar1'][0]*beta_t['beta_2'] + \
                        self.ts['beta_2']['ar2'][0]*beta_t2['beta_2'] + \
                        self.ts['beta_2']['mxreg1'][0]*beta_t['beta_3'] + \
                        self.ts['beta_2']['mxreg2'][0]*beta_t['beta_5'] + \
                        np.sqrt(h_beta_2_t_1)*epsilon_beta_2
        
        epsilon_beta_3 = nig_innovation(innov_cdf[3], self.ts['beta_3']['skew'][0], self.ts['beta_3']['shape'][0])
        h_beta_3_t_1 =  self.ts['beta_3']['omega'][0] + \
                        self.ts['beta_3']['alpha1'][0]*h_t['h_beta_3']*np.power(e_t['beta_3_residuals'] - self.ts['beta_3']['eta21'][0], 2)
        beta_3_t_1 = self.ts['beta_3']['mu'][0] + \
                        self.ts['beta_3']['ar1'][0]*beta_t['beta_3'] + \
                        self.ts['beta_3']['mxreg1'][0]*beta_t['beta_2'] + \
                        self.ts['beta_3']['mxreg2'][0]*beta_t['beta_4'] + \
                        self.ts['beta_3']['mxreg3'][0]*beta_t['beta_5'] + \
                        np.sqrt(h_beta_3_t_1)*epsilon_beta_3

        epsilon_beta_4 = nig_innovation(innov_cdf[4], self.ts['beta_4']['skew'][0], self.ts['beta_4']['shape'][0])
        h_beta_4_t_1 =  self.ts['beta_4']['omega'][0] + \
                        self.ts['beta_4']['alpha1'][0]*h_t['h_beta_4']*np.power(e_t['beta_4_residuals'] - self.ts['beta_4']['eta21'][0], 2)
        beta_4_t_1 = self.ts['beta_4']['mu'][0] + \
                        self.ts['beta_4']['ar1'][0]*beta_t['beta_4'] + \
                        self.ts['beta_4']['mxreg1'][0]*beta_t['beta_3'] + \
                        np.sqrt(h_beta_4_t_1)*epsilon_beta_4

        epsilon_beta_5 = nig_innovation(innov_cdf[5], self.ts['beta_5']['skew'][0], self.ts['beta_5']['shape'][0])
        h_beta_5_t_1 =  self.ts['beta_5']['omega'][0] + \
                        self.ts['beta_5']['vxreg1'][0]*np.power(self.beta_std['beta_5'], 2) + \
                        self.ts['beta_5']['alpha1'][0]*h_t['h_beta_5']*np.power(e_t['beta_5_residuals'] - self.ts['beta_5']['eta21'][0], 2)
        beta_5_t_1 = self.ts['beta_5']['mu'][0] + \
                        self.ts['beta_5']['ar1'][0]*beta_t['beta_5'] + \
                        self.ts['beta_5']['ar2'][0]*beta_t2['beta_5'] + \
                        self.ts['beta_5']['mxreg1'][0]*beta_t['beta_1'] + \
                        self.ts['beta_5']['mxreg2'][0]*beta_t['beta_4'] + \
                        np.sqrt(h_beta_5_t_1)*epsilon_beta_5

        epsilon_Y = nig_innovation(innov_cdf[0], self.ts['y']['skew'][0], self.ts['y']['shape'][0])
        h_Y_t_1 = self.ts['y']['vxreg1'][0]*np.power(ATM_1mo_IV_t, 2)
        Y_t_1 = self.ts['y']['mxreg1'][0]*(r_t_tau-q_t) + np.sqrt(h_Y_t_1)*epsilon_Y

        S_t_1 = S_t*np.exp(Y_t_1)
        beta_t_1 = np.array([beta_1_t_1, beta_2_t_1, beta_3_t_1, beta_4_t_1, beta_5_t_1])
        h_t_1 = np.array([h_Y_t_1, h_beta_1_t_1, h_beta_2_t_1, h_beta_3_t_1, h_beta_4_t_1, h_beta_5_t_1])
        e_t_1 = np.array([epsilon_Y, epsilon_beta_1, epsilon_beta_2, epsilon_beta_3, epsilon_beta_4, epsilon_beta_5])
        return S_t_1, beta_t_1, h_t_1, e_t_1, q_t, r_t_tau

    def fit(self, sigma, S, r, q, tau, K):
        beta_t = np.empty((sigma.index.nunique(), 5))
        unique_time = sigma.index.sort_values().unique()
        for i, t in enumerate(unique_time):
            beta_t[i, :] = np.squeeze(self._fit_for_t(sigma[t], S[t], r[t], q[t], tau[t], K[t]))
        self.beta = pd.DataFrame(data={'beta_1':beta_t[:, 0], 'beta_2':beta_t[:, 1], 'beta_3':beta_t[:, 2], 'beta_4':beta_t[:, 3], 'beta_5':beta_t[:, 4]}, index=unique_time)
        self._adjust_params()
        self.S = S
        self.r = r
        self.q = q

    def _fit_for_t(self, sigma, S, r, q, tau, K):
        n = sigma.size
        sigma_t = sigma.to_numpy().reshape((n, 1))
        S_t = S.to_numpy().reshape((n, 1))
        r_t = r.to_numpy().reshape((n, 1))
        q_t = q.to_numpy().reshape((n, 1))
        tau_t = tau.to_numpy().reshape((n, 1))
        K_t = K.to_numpy().reshape((n, 1))
        F_t_tau = forward_price(S_t, r_t, q_t, tau_t)
        M_t = moneyness(tau_t, F_t_tau, K_t)

        Y = sigma_t
        X = np.concatenate([np.ones((n, 1)), \
                            time_to_maturity(tau_t, self.T_conv), \
                            moneyness_slope(M_t), \
                            smile_attenuation(M_t, tau_t, self.T_max), \
                            smirk(M_t, tau_t, self.T_max)], axis=1)
        beta_t = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        return beta_t
    
    def _adjust_params(self):
        std_5 = np.std(self.beta.iloc[:, 4])
        self.beta.iloc[:, 4] = np.where(self.beta.iloc[:, 4] > std_5, np.nan, self.beta.iloc[:, 4])
        self.beta.iloc[:, 4] = np.where(self.beta.iloc[:, 4] < -std_5, np.nan, self.beta.iloc[:, 4])
        mean_5 = np.mean(self.beta.iloc[:, 4])
        self.beta.iloc[:, 4] = np.where(pd.isna(self.beta.iloc[:, 4]), mean_5, self.beta.iloc[:, 4])

    def save(self, filepath):
        self.beta.to_csv(filepath, index=True)