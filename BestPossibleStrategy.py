import numpy as np
import pandas as pd
from util import get_data
from marketsimcode import compute_portvals, cal_port_stats
from ManualStrategy import plot_compare
import datetime as dt

#BestPossibleStrategy
def testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
    df = get_data([symbol], dates = pd.date_range(sd, ed))

    ind = []
    current = 0
    for i in range(len(df.index)-1):
        if df[symbol][i] > df[symbol][i + 1]:
            if current == 0:
                ind.append([str(df.index[i]).split()[0], -1000])
                current = -1000
            elif current == 1000:
                ind.append([str(df.index[i]).split()[0], -2000])
                current = -1000
            else:
                ind.append([str(df.index[i]).split()[0], 0])
                current = -1000
        elif df[symbol][i] < df[symbol][i + 1]:
            if current == 0:
                ind.append([str(df.index[i]).split()[0], 1000])
                current = 1000
            elif current == -1000:
                ind.append([str(df.index[i]).split()[0], 2000])
                current = 1000
            else:
                ind.append([str(df.index[i]).split()[0], 0])
                current = 1000
        else:
            ind.append([str(df.index[i]).split()[0], 0])
    
    #print df[symbol]
    #print ind
    ordersfile = open('bps_orders_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]), 'w')
    ordersfile.write("Date,Symbol,Order,Shares\n")
    for i in range(len(ind)):
        if ind[i][1] == 1000:
            ordersfile.write("%s,%s,BUY,1000\n"%(ind[i][0],symbol))
        elif ind[i][1] == -1000:
            ordersfile.write("%s,%s,SELL,1000\n"%(ind[i][0],symbol))
        elif ind[i][1] == 2000:
            ordersfile.write("%s,%s,BUY,2000\n"%(ind[i][0],symbol))
        elif ind[i][1] == -2000:
            ordersfile.write("%s,%s,SELL,2000\n"%(ind[i][0],symbol))

    ordersfile.close()


def test():
	sd = dt.datetime(2008,1,1)
	ed = dt.datetime(2009,12,31)
	df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
	portvals = compute_portvals(orders_file = 'bps_orders_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]))
	portvals = portvals[portvals.columns[0]]
	plot_compare(df=portvals, symbol = "JPM", sd = sd, ed = ed, title = 'BPS_{}_{}'.format(str(sd).split()[0], str(ed).split()[0]))

	start_date = portvals.index[0]
	end_date = portvals.index[-1]
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)

	print "Date Range: {} to {}".format(start_date, end_date)

	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	#print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)

	print "Cumulative Return of Fund: {}".format(cum_ret)
	#print "Cumulative Return of SPY : {}".format(cum_ret_SPY)

	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	#print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)

	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	#print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)

	print "Final Portfolio Value: {}".format(portvals[-1])

	portvals = compute_portvals(orders_file = 'benchmark_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]))
	portvals = portvals[portvals.columns[0]]

	start_date = portvals.index[0]
	end_date = portvals.index[-1]
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)

	print "benchmark"
	print "Date Range: {} to {}".format(start_date, end_date)

	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	#print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)

	print "Cumulative Return of Fund: {}".format(cum_ret)
	#print "Cumulative Return of SPY : {}".format(cum_ret_SPY)

	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	#print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)

	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	#print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)

	print "Final Portfolio Value: {}".format(portvals[-1])

	sd = dt.datetime(2010,1,1)
	ed = dt.datetime(2011,12,31)
	df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
	portvals = compute_portvals(orders_file = 'bps_orders_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]))
	portvals = portvals[portvals.columns[0]]
	plot_compare(df=portvals, symbol = "JPM", sd = sd, ed = ed, title = 'BPS_{}_{}'.format(str(sd).split()[0], str(ed).split()[0]))

	start_date = portvals.index[0]
	end_date = portvals.index[-1]
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)

	print "Date Range: {} to {}".format(start_date, end_date)

	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	#print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)

	print "Cumulative Return of Fund: {}".format(cum_ret)
	#print "Cumulative Return of SPY : {}".format(cum_ret_SPY)

	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	#print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)

	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	#print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)

	print "Final Portfolio Value: {}".format(portvals[-1])

	portvals = compute_portvals(orders_file = 'benchmark_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]))
	portvals = portvals[portvals.columns[0]]

	start_date = portvals.index[0]
	end_date = portvals.index[-1]
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)

	print "benchmark"
	print "Date Range: {} to {}".format(start_date, end_date)

	print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
	#print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)

	print "Cumulative Return of Fund: {}".format(cum_ret)
	#print "Cumulative Return of SPY : {}".format(cum_ret_SPY)

	print "Standard Deviation of Fund: {}".format(std_daily_ret)
	#print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)

	print "Average Daily Return of Fund: {}".format(avg_daily_ret)
	#print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)

	print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == '__main__':
	test()

