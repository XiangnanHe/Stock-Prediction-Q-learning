#Name: Xiangnan He
#ID: xhe321

"""
Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "order.csv", start_val = 100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    trade_df = pd.read_csv(orders_file, index_col = 'Date')

    start_date =  trade_df.index[0]
    end_date = trade_df.index[-1]
    sym_list = trade_df.Symbol.unique().tolist()
    dates = trade_df.index.unique().tolist()
    price_df = get_data(sym_list, pd.date_range(start_date, end_date), addSPY = True)
    trade_df = trade_df.join(price_df, how = 'right') 
    trade_df = trade_df.drop('SPY', 1)
    for sym in sym_list:    
        trade_df.ix[:,sym] = 0.0

    trade_df.ix[0,'cash'] = float(start_val)    
    trade_df['total'] = 0.0

    for i in range(trade_df.shape[0]):
        stock = trade_df.Symbol[i]
        if pd.isnull(stock):
            trade_df.ix[i, 'comsn'] = 0.0
            trade_df.ix[i, 'cash'] = trade_df.ix[i-1, 'cash']
            trade_df.ix[i, 'trade_val'] = 0.0
            trade_df.ix[i, 'total'] = trade_df.ix[i, 'total'] + trade_df.ix[i, 'cash']
            for sym in sym_list:    
                trade_df.ix[i, sym] = trade_df.ix[i - 1, sym] 
                trade_df.ix[i, 'total'] = trade_df.ix[i, 'total'] + trade_df.ix[i, sym] * price_df.ix[trade_df.index[i], sym]        
            continue

        shares = float(trade_df.Shares[i])
        if trade_df.Order[i] == 'BUY':
            order = 1.0
        else: 
            order = -1.0

        date_i = trade_df.index[i]
        price = float(price_df[stock][date_i])    
        trade_val = price*shares*order
        trade_df.ix[i, 'trade_val'] = trade_val
        trade_df.ix[i, 'comsn'] = commission
        
        if i == 0:
            trade_df.ix[i, 'cash'] = float(trade_df.ix[i, 'cash']) - trade_val - commission - abs(trade_val)*impact 
            trade_df.ix[i, stock] = shares * order
            trade_df.ix[i, 'total'] = float(start_val) 
        else:
            for sym in sym_list:    
                trade_df.ix[i, sym] = trade_df.ix[i - 1, sym]  
                
            #consider the case when the trade involves 0 shares
            if shares > 0:
                trade_df.ix[i, 'cash'] = trade_df.ix[i-1, 'cash'] - trade_val - commission- abs(trade_val)*impact 
            elif shares == 0:
                trade_df.ix[i, 'cash'] = trade_df.ix[i-1, 'cash']
            trade_df.ix[i, stock] = trade_df.ix[i-1, stock] + shares * order
            trade_df.ix[i, 'total'] = trade_df.ix[i, 'total'] + trade_df.ix[i, 'cash']
            
            for sym in sym_list: 
                trade_df.ix[i, 'total'] = trade_df.ix[i, 'total'] + trade_df.ix[i, sym] * price_df.ix[trade_df.index[i], sym]
    
    portvals = trade_df.total  # remove SPY
    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())
    rv = rv.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    return rv

def compute_portvals_sl(orders, sym='JPM', start_val=100000, commission=0.0, impact=0.0):

    # Read orders file
    orders_df = orders

    orders_df.sort_index(inplace=True)
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]

    # Collect price data for each ticker in order
    df_prices = get_data([sym], pd.date_range(start_date, end_date))
    df_prices = df_prices.drop('SPY', 1)  # remove SPY
    df_prices['cash'] = 1

    # Track trade data
    df_trades = df_prices.copy()
    df_trades[:] = 0

    # Populate trade dataframe
    for i, date in enumerate(orders_df.index):
        shares = orders_df['Shares'][date]

        # Calculate change in shares and cash
        df_trades[sym][date] +=  shares
        df_trades['cash'][date] -=  (1 - impact) * shares * df_prices[sym][date] - commission

    # Track total holdings
    df_holdings = df_prices.copy()
    df_holdings[:] = 0

    # Include starting value
    df_holdings['cash'][0] = start_val

    # Update first day of holdings
    for c in df_trades.columns:
        df_holdings[c][0] += df_trades[c][0]

    # Update every day, adding new day's trade information with previous day's holdings
    for i in range(1, len(df_trades.index)):
        for c in df_trades.columns:
            df_holdings[c][i] += df_trades[c][i] + df_holdings[c][i - 1]

    # Track monetary values
    df_values = df_prices.mul(df_holdings)

    # Define port_val
    port_val = df_values.sum(axis=1)

    return port_val

def author():
    return 'xhe321' # replace tb34 with your Georgia Tech username.

def cal_port_stats(port_val, rfr, sf):
    daily_rets = port_val.copy()
    daily_rets = (port_val/port_val.shift(1)) - 1
    adr = daily_rets.mean()
    cr = (port_val[-1]/port_val[0]) - 1
    sddr = daily_rets.std()
    k = np.sqrt(sf)    
    sr = k * np.mean(daily_rets - rfr)/sddr   
    return cr,adr,sddr,sr 

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    portvals_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    portvals_SPY = portvals_SPY['SPY']

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = cal_port_stats(portvals_SPY, 0.0, 252.0)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
