#Name: Xiangnan He
#ID: xhe321

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime as dt 
import util
import StrategyLearner as sl
import ManualStrategy as ms 
from marketsimcode import compute_portvals as cp
from marketsimcode import compute_portvals_sl as cp_sl

def cal_port_stats(port_val, rfr, sf):
    daily_rets = port_val.copy()
    daily_rets = (port_val/port_val.shift(1)) - 1
    adr = daily_rets.mean()
    cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    sddr = daily_rets.std()
    k = np.sqrt(sf)    
    sr = k * np.mean(daily_rets - rfr)/sddr   
    return cr,adr,sddr,sr 


def experiment1():
	symbol = 'JPM'
	sv = 100000
	verbose = False
	impact = 0.0

	# in sample
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)

	s_learner = sl.StrategyLearner(verbose = verbose, impact = impact)

	t0 = time.clock()
	s_learner.addEvidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
	t1 = time.clock()

	df= s_learner.testPolicy(symbol, sd, ed, sv)
	t2 = time.clock()

	print "Experiment1 in sample:"
	print "Time for addEvidence: {}\n".format(t1-t0)
	print "Time for testPolicy: {}\n".format(t2-t1)
	print "Date Range: {} to {}".format(sd, ed)
	#print df
	#price_df = sl.Stock(symbol = symbol, sd=sd, ed=ed,sv=sv, verbose=verbose)
	#price_df = price_df.df

	#print price_df
	#portvals = cp(orders_file = 'QLearner_order_{}_{}.csv'.format(str(sd).split()[0],str(ed).split()[0]), 
	#	start_val = sv, commission = 0.0, impact = impact)

	start = df.index[0]
	end = df.index[-1]
	benchmark = pd.DataFrame({'Shares':[1000, -1000]}, index = [start, end])
	benchmark = cp_sl(orders = benchmark)
	
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(benchmark, 0.0, 252.0)
	benchmark = benchmark/benchmark.iloc[0]
	print "Benchmark:"
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Final Portfolio Value: {}".format(benchmark[-1]*sv)

	portvals_ms = ms.testPolicy(symbol = symbol, sd = sd, ed = ed,
		sv = sv)
	#print portvals_ms
	benchmark = pd.DataFrame(benchmark)
	portvals_ms = portvals_ms.join(benchmark, lsuffix = '_1', rsuffix = '_2', how = 'right')
	portvals_ms = pd.DataFrame(portvals_ms['0_1'])
	portvals_ms.fillna(method = 'ffill', inplace=True)
	portvals_ms.fillna(method = 'bfill', inplace = True)
	#portvals_ms = pd.DataFrame(portvals_ms['0_1'])
	
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals_ms, 0.0, 252.0)
	portvals_ms=portvals_ms/portvals_ms.iloc[0]
	print "ManualStrategy:"
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Final Portfolio Value: {}".format(portvals_ms.iloc[-1]*sv)


	portvals = cp_sl(orders = df)
	

	portvals.fillna(method = 'ffill', inplace=True)
	portvals.fillna(method = 'bfill', inplace = True)
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)
	portvals=portvals/portvals.iloc[0]
	print "Qlearner:"
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Final Portfolio Value: {}".format(portvals[-1]*sv)

	benchmark.columns = ['Benchmark']
	portvals_ms.columns = ['ManualStrategy']
	ax = benchmark.plot(title = 'Experiment1-in sample', label = 'Benchmark', color = 'black')
	portvals.plot(label = 'QLearner', ax=ax, color = 'blue')
	portvals_ms.plot(label = 'Manual', ax = ax, color = 'green')

	ax.legend(loc='best')
	plt.savefig('QLearner_{}_{}.png'.format(str(sd).split()[0], str(ed).split()[0]))
	plt.close('all')
###########################################################################################
	# out of sample
	sd = dt.datetime(2010, 1, 1)
	ed = dt.datetime(2011, 12, 31)

	#s_learner = sl.StrategyLearner(verbose = verbose, impact = impact)

	#t0 = time.clock()
	#s_learner.addEvidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
	#t1 = time.clock()

	df= s_learner.testPolicy(symbol, sd, ed, sv)
	#t2 = time.clock()

	print "Experiment1 Out Of Sample:"
	print "Time for addEvidence: {}\n".format(t1-t0)
	print "Time for testPolicy: {}\n".format(t2-t1)
	print "Date Range: {} to {}".format(sd, ed)

	start = df.index[0]
	end = df.index[-1]
	benchmark = pd.DataFrame({'Shares':[1000, -1000]}, index = [start, end])
	benchmark = cp_sl(orders = benchmark)
	
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(benchmark, 0.0, 252.0)
	benchmark = benchmark/benchmark.iloc[0]
	print "Benchmark:"
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Final Portfolio Value: {}".format(benchmark[-1]*sv)

	portvals_ms = ms.testPolicy(symbol = symbol, sd = sd, ed = ed,
		sv = sv)
	#print portvals_ms
	benchmark = pd.DataFrame(benchmark)
	portvals_ms = portvals_ms.join(benchmark, lsuffix = '_1', rsuffix = '_2', how = 'right')

	portvals_ms = pd.DataFrame(portvals_ms['0_1'])
	portvals_ms.fillna(method = 'ffill', inplace=True)
	portvals_ms.fillna(method = 'bfill', inplace = True)
	
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals_ms, 0.0, 252.0)
	portvals_ms=portvals_ms/portvals_ms.iloc[0]
	print "ManualStrategy:"
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Final Portfolio Value: {}".format(portvals_ms.iloc[-1]*sv)


	portvals = cp_sl(orders = df)
	

	portvals.fillna(method = 'ffill', inplace=True)
	portvals.fillna(method = 'bfill', inplace = True)
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)
	portvals=portvals/portvals.iloc[0]
	print "Qlearner:"
	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	print "Cumulative Return of Fund: {}".format(cum_ret)
	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	print "Final Portfolio Value: {}".format(portvals[-1]*sv)

	benchmark.columns = ['Benchmark']
	portvals_ms.columns = ['ManualStrategy']
	ax = benchmark.plot(title = 'Experiment1-out of sample', label = 'Benchmark', color = 'black')
	portvals.plot(label = 'QLearner', ax=ax, color = 'blue')
	portvals_ms.plot(label = 'Manual', ax = ax, color = 'green')

	ax.legend(loc='best')
	plt.savefig('QLearner_{}_{}.png'.format(str(sd).split()[0], str(ed).split()[0]))
	plt.close('all')

if __name__=="__main__":
	experiment1()


