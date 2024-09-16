#!/usr/bin/env python
# coding: utf-8

import re
import os
import math
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

def load_data_(file_path, as_of_date):
    d = pd.read_excel(file_path)  #read the file into memory NOTE: its an excel file
    #finding index & the value of merged cells in data Eg: 17-Mar-23 (24d); CSize 100
    idx = []
    idx_c = []
    for i,j in enumerate(d['Strike']):
        if type('string') == type(j):
            #print(i,j)
            idx.append(i)
            idx_c.append(j)
    #get the values between two merged cells (the data we need)        
    diff = np.diff(idx)
    diff = list(diff)
    diff.append(len(d) - idx[len(idx)-1])
    diff = [x - 1 for x in diff]
    #columns to add
    exp_date = []
    as_of = []
    time_to_maturity = []
    #getting the col values
    for i, j in zip(idx_c,diff):
        exp_date.append([datetime.datetime.strptime(i.split(' ')[0],"%d-%b-%y")]*j)
        as_of.append([datetime.datetime.strptime(as_of_date,"%d-%b-%y")]*j)
        time_to_maturity.append([int(re.findall(r'\d+',i.split(' ')[1])[0])]*j)
    #dropping the merged cells
    d_new = d.drop(idx)
    #appending the new columns
    d_new['exp_date'] = [item for sublist in exp_date for item in sublist]
    d_new['as_of'] = [item for sublist in as_of for item in sublist]
    d_new['given_time_to_maturity'] = [item for sublist in time_to_maturity for item in sublist]
    #finding the actual time to maturity by finding business days 
    test = []
    for as_of_, exp_date_ in zip(d_new['as_of'],d_new['exp_date']):
        test.append(len(pd.bdate_range(as_of_, exp_date_).drop(as_of_)))
        
    d_new['time_to_maturity'] = test
    
    return d_new

def get_rates(years=[2022,2023]):
    comb = []
    #for each year it will get the data from treasury.gov
    for year in years:
        df_ = pd.read_html('https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={}'.format(year))
        comb.append(df_[0].drop(df_[0].columns[df_[0].isna().any()].tolist(), axis=1))
    ret = comb[0]
    #combine the data into single dataset (as we will have multiple years data to pull)
    for each in range(1,len(comb)):
        ret = ret.append(comb[each], ignore_index=True)
    ret['Date'] = pd.to_datetime(ret['Date'])    
    return ret    


#getting the rates for once and use it later on, this will save a lot of execution time
rts = get_rates()
rts.to_csv('./rates.csv')

def closest(lst, K):
    #finding the closest value to K from the elements of the list lst
    return lst.index(lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))])


def update_rates(dataframe, r=rts):
    rates_df = r
    #list of number of days to expiry (bonds have 1mo, 2mo, 3mo ...)
    #84 means 4mo, its at the end because data we pulled gives it at the end
    rt_lst = [21, 42, 63, 126, 252, 504, 756, 1260, 1764, 2520, 5040, 7560, 84]
    rts = []
    for as_of_, exp_date_, ttm_ in zip(dataframe['as_of'], dataframe['exp_date'], dataframe['time_to_maturity']):
        #del_ = int((exp_date_ - as_of_)/np.timedelta64(1, 'D'))
        #print(del_,ttm_)
        col_ = closest(rt_lst, ttm_)
        if math.isnan(rates_df.loc[rates_df['Date'] == as_of_].iloc[:, col_+1].values[0]):
            rate_ = rates_df.loc[rates_df['Date'] == as_of_].iloc[:, 4].values[0]
        else:
            rate_ = rates_df.loc[rates_df['Date'] == as_of_].iloc[:, col_+1].values[0]
        #print(ttm_)
        #print(rate_/100)
        #print(np.log((1+(rate_/100))**(ttm_/(12*21)))/ttm_)
        rts.append(np.log((1+(rate_/100))**(ttm_/(12*21)))/ttm_)
    
    dataframe['r'] = rts
    
    #return dataframe


def update_dividends(dataframe, file_path, as_of_date):
    dividend_df = pd.read_csv(file_path) #reading the dividend yeild from the file
    dividend_df['Date'] = pd.to_datetime(dividend_df['Date'])
    
    dataframe['q'] = [np.log((1+(dividend_df.loc[dividend_df['Date'] == as_of_date]['12M_DVD_YLD'].values[0]/100)))/252]*len(dataframe)
    

def get_data(file_path_base, as_of_date, file_path_dividend):
    #, strike_price_size=5
    final_df = load_data_(file_path=file_path_base,
         as_of_date=as_of_date) #strike_price_size=strike_price_size
    
    update_rates(dataframe=final_df)
    
    update_dividends(dataframe=final_df, 
                 file_path=file_path_dividend, 
                 as_of_date=as_of_date)
    
    return final_df

def get_date_str(str_):
    date_obj = datetime.datetime.strptime(str_, '%Y-%m-%d')
    return datetime.datetime.strftime(date_obj, '%d-%b-%y')

def get_combined_data(data_dir='./data',output_file='./db.csv'):
    directory = './data'
    files = []
    for filename in os.scandir(directory):
        if filename.is_file():
            files.append(filename.path)
    files.remove('./data/.DS_Store')
    final_ = pd.DataFrame()
    for file in files:
        #print(file) #uncomment to debug on which file we get error
        asOf = get_date_str('-'.join(file.split('_')[2:5]))
        data = get_data(file_path_base=file,
                   as_of_date=asOf,
                   file_path_dividend='./spy_12mdvdyld.csv') #strike_price_size=4
        final_ = final_.append(data, ignore_index=True)
    data = yf.download("SPY", progress=False, interval='1d', period='max')
    data.reset_index(inplace=True)
    final_['as_of_stock_price'] = [data.loc[data['Date'] == dt]['Close'].values[0] for dt in final_['as_of']]
    final_.to_csv(output_file)
    return final_


fd = get_combined_data()




