#Name: Xiangnan He
#ID: xhe321

import numpy as np
import pandas as pd
from indicators import bol_bands, SMA, momentum
import datetime as dt
from util import get_data
from marketsimcode import compute_portvals, cal_port_stats
import matplotlib.pyplot as plt

def gen_benchmark(symbol = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31)):
	ordersfile = open('benchmark_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]), 'w')
	ordersfile.write("Date,Symbol,Order,Shares\n")

	price = get_data([symbol], pd.date_range(sd, ed))
	price = price[symbol]/price[symbol][0]
	price = pd.DataFrame(index=price.index, data=price.as_matrix(), columns = [symbol])
	#print price
	
	dates = [str(price.index[0]).split()[0],str(price.index[-1]).split()[0]]
	ordersfile.write("%s,%s,BUY,1000\n"%(dates[0],symbol))
	ordersfile.write("%s,%s,SELL,0\n"%(dates[1],symbol))

	ordersfile.close()	


def plot_compare(df, symbol = "JPM", sd = '2008-01-01', ed = '2009-12-31', long_entries = [], long_exits=[], short_entries = [], short_exits = [], title = "Stock prices"):
    df = df/df[0]
    rv = pd.DataFrame(index=df.index, data=df.as_matrix(), columns = ['Profile'])
    price = get_data([symbol], pd.date_range(sd,ed))
    price1 = price[symbol]/price[symbol][0]
    price1 = pd.DataFrame(index=price1.index, data=price1.as_matrix(), columns = [symbol]) 

    benchmark = compute_portvals(orders_file = 'benchmark_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]))
    
    benchmark = benchmark[benchmark.columns[0]]
    benchmark = benchmark/benchmark[0]
    benchmark = pd.DataFrame(index=benchmark.index, data=benchmark.as_matrix(), columns = ['benchmark'])
    

    rv = price1.join(rv) 
    rv = rv.join(benchmark)

    colors = ['orange', 'black', 'blue']

    #print rv

    rv.fillna(method='ffill', inplace = True)
    rv.fillna(method='bfill', inplace = True)
    
    xlabel = "Date"
    ylabel = "Price"

    ax = rv.plot(title=title, fontsize=12, color = colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ymin,ymax = ax.get_ylim()
    plt.vlines(long_entries,ymin,ymax,color='g')
    plt.vlines(long_exits,ymin,ymax)
    plt.vlines(short_entries,ymin,ymax,color='r')
    plt.vlines(short_exits,ymin,ymax)

    plt.savefig('{}_{}_{}.png'.format(title, str(sd).split()[0], str(ed).split()[0]))
    plt.close('all')
    #plt.show()

def combined_ind(df, symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), window = 10):
	mome10 = momentum(df, symbol, window=10)[1]
	SMA20,div20 = SMA(df, symbol, window=20)[1:]
	bb_ind20 = bol_bands(df, symbol, window = 20)[3]
	#print mome10
	mome10 = mome10/mome10.iloc[10]
	div20 = div20/div20.iloc[19]
	bb_ind20 = bb_ind20/bb_ind20.iloc[19]
	#print momentum1

	short_entries = []
	short_exits = []
	long_entries = []
	long_exits = []
	signals = []

	entry = False


	for i in range(len(df.index)):
	    if mome10[i-1] > 3.0 and mome10[i] < 2.8  and div20[i] <1.5 and bb_ind20[i] < 1.0 and not entry:
	        short_entries.append(df.index[i])
	        entry = True
	        signals.append([str(df.index[i]).split()[0], 'BUY'])
	    if mome10[i-1] < -2.0  and div20[i] > 0.5 and bb_ind20[i] > -1.0 and entry:
	        short_exits.append(df.index[i])
	        entry = False
	        signals.append([str(df.index[i]).split()[0], 'SELL'])
	    if mome10[i-1] <-6.0 and mome10[i] >-6.0 and div20[i] <1.5 and bb_ind20[i] < 1.0 and not entry:
	        long_entries.append(df.index[i])
	        entry = True
	        signals.append([str(df.index[i]).split()[0], 'BUY'])
	    if mome10[i-1] > 5.0 and mome10[i-1] < 5.0 and div20[i] >0.5 and bb_ind20[i] > -1.0 and entry:
	        long_exits.append(df.index[i])
	        entry = False
	        signals.append([str(df.index[i]).split()[0], 'SELL'])

	ordersfile = open('combined_indicator_order_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]), 'w')
	ordersfile.write("Date,Symbol,Order,Shares\n")

	for signal in signals:
	    ordersfile.write("%s,%s,%s,1000\n"%(signal[0],symbol,signal[1]))

	ordersfile.close()
	df[symbol] = df[symbol]/df[symbol].iloc[0]

	ax = df[symbol].plot(title = 'Combined_ind', label = symbol)
	mome10.plot(label = 'Mome10', ax = ax)
	div20.plot(label = 'Div20', ax = ax)
	bb_ind20.plot(label = 'bb_ind20', ax=ax)


	ymin, ymax = ax.get_ylim()
	plt.vlines(long_entries,ymin,ymax,color='g')
	plt.vlines(long_exits,ymin,ymax)
	plt.vlines(short_entries,ymin,ymax,color='r')
	plt.vlines(short_exits,ymin,ymax)

	ax.legend(loc='lower right')
	plt.savefig('combined_ind_{}_{}.png'.format(str(sd).split()[0], str(ed).split()[0]))
	plt.close('all')
	#plt.show()
	return long_entries,long_exits, short_entries, short_exits

def testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):


	dates = pd.date_range(sd,ed)
	df = get_data([symbol],dates)

	long_entries,long_exits, short_entries, short_exits = combined_ind(df, "JPM", sd, ed, window = 20)
	portvals = compute_portvals(orders_file = 'combined_indicator_order_{}_{}.csv'.format(str(sd).split()[0], str(ed).split()[0]))

	portvals = portvals[portvals.columns[0]]

	start_date = portvals.index[0]
	end_date = portvals.index[-1]
	cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals, 0.0, 252.0)

	portvals = pd.DataFrame(portvals)

	return portvals

def test(symbol = 'JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
	return testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv) 

	#df_trades = testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) 

if __name__ == '__main__':
	test()


