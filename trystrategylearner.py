"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import util as ut
import StrategyLearner as sl
import time
import warnings

from matplotlib import style
from util import get_data

style.use('ggplot')
warnings.filterwarnings('ignore')


def test_code(sym, c0, insample=False, verb=False):
    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose=verb)

    # set parameters for training the learne
    stdate = dt.datetime(2008, 1, 1)
    enddate = dt.datetime(2009, 12, 31)  # just a few days for "shake out"

    # train the learner
    t0 = time.clock()
    learner.addEvidence(symbol=sym, sd=stdate,
                        ed=enddate, sv=c0)
    t1 = time.clock()

    if insample:
        print "Time to complete addEvidence():\t\t\t\t\t{:.1f} sec".format(t1 - t0)

    # set parameters for testing
    if not insample:
        stdate = dt.datetime(2010, 1, 1)
        enddate = dt.datetime(2011, 12, 31)

    # get some data for reference
    syms = [sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb:
        print prices

    # test the learner
    t2 = time.clock()
    df_trades = learner.testPolicy(symbol=sym, sd=stdate,
                                   ed=enddate, sv=c0)
    t3 = time.clock()

    if insample:
        sample = 'in-sample:\t\t'
    else:
        sample = 'out-of-sample:\t'

    print "Time to complete testsample() {}{:.1f} sec".format(sample, t3 - t2)

    if verb:
        print df_trades

    return df_trades


def compute_portvals(sym, orders, start_val):
    fee = 5  # $5 trade comission fee
    slip = 0.00025  # 2.5 bips slippage

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
        # Get order information
        if orders_df.Order[i] == 'BUY':
            order = 1
        else:
            order = -1

        if i == 0:
            shares = 200
        else:
            shares = 400

        # Calculate change in shares and cash
        df_trades[sym][date] += order * shares
        df_trades['cash'][date] -= order * (1 - slip) * shares * df_prices[sym][date] - fee

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

"""
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
"""

if __name__ == "__main__":
    # Settings
    # ------------------------------------------------------- #
    case = ['ML4T-220', 'AAPL', 'SINE_FAST_NOISE', 'UNH', 'JPM']
    symbol = case[3]
    verbose = False
    c0 = 100000
    # ------------------------------------------------------- #

    df_trades_in, benchmark_in = test_code(symbol, c0, True, verbose)
    df_trades_out, benchmark_out = test_code(symbol, c0, False, verbose)

    # Evaluate Results
    port_in = compute_portvals(symbol, df_trades_in, c0)
    port_in = pd.DataFrame(port_in)

    bench_in = compute_portvals(symbol, benchmark_in, c0)
    bench_in = pd.DataFrame(bench_in)

    port_out = compute_portvals(symbol, df_trades_out, c0)
    port_out = pd.DataFrame(port_out)

    bench_out = compute_portvals(symbol, benchmark_out, c0)
    bench_out = pd.DataFrame(bench_out)

    # Calculate cumulative returns
    port_ret_in = float(np.asarray(port_in.values)[-1])
    port_ret_out = float(np.asarray(port_out.values)[-1])
    bench_ret_in = float(np.asarray(bench_in.values)[-1])
    bench_ret_out = float(np.asarray(bench_out.values)[-1])

    # Print results
    print
    print 'Cumulative return in-sample:\t\t${:,.2f}\t\t(+{:.2f} %)'.format(port_ret_in - c0,
                                                                           100 * (port_ret_in - c0) / c0)
    print 'Benchmark return in-sample:\t\t\t${:,.2f}\t\t(+{:.2f} %)'.format(bench_ret_in - c0,
                                                                            100 * (bench_ret_in - c0) / c0)
    print 'Cumulative return out-of-sample:\t${:,.2f}\t\t(+{:.2f} %)'.format(port_ret_out - c0,
                                                                             100 * (port_ret_out - c0) / c0)
    print 'Benchmark return out-of-sample:\t\t${:,.2f}\t\t(+{:.2f} %)'.format(bench_ret_out - c0,
                                                                              100 * (bench_ret_out - c0) / c0)

    plt.subplot(1, 2, 1)
    plt.plot(port_in.index, port_in, c='lightcoral')
    plt.plot(bench_in.index, bench_in, c='skyblue')
    plt.legend(['Algorithmic Strategy', 'Buy and Hold'])
    plt.title('Q-Learner Algorithmic Trader In-Sample')
    plt.xlabel('Date')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    plt.plot(port_out.index, port_out, c='mediumseagreen')
    plt.plot(bench_out.index, bench_out, c='skyblue')
    plt.legend(['Algorithmic Strategy', 'Buy and Hold'])
    plt.title('Q-Learner Algorithmic Trader Out-of-Sample')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
