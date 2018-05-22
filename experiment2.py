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

def experiment2(impact = 0.0):
	symbol = 'JPM'
	sv = 100000
	verbose = False
	

	# in sample
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	#trade_df=pd.DataFrame()
	#for i in [0.0, 0.02, 0.04, 0.08]:
	s_learner = sl.StrategyLearner(verbose = verbose, impact = impact)
	#t0 = time.clock()
	s_learner.addEvidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
	#t1 = time.clock()
	df= s_learner.testPolicy(symbol, sd, ed, sv)
	#df.columns = [str(i)]
	#t2 = time.clock()

	portvals = cp_sl(orders = df)
	portvals = pd.DataFrame(portvals)
	portvals.fillna(method = 'ffill', inplace=True)
	portvals.fillna(method = 'bfill', inplace = True)
	portvals=portvals/portvals.iloc[0]
	portvals.columns = [str(impact)]
	portvals_in = portvals
	trade_num_in = len(df)

	# out of sample
	sd = dt.datetime(2010, 1, 1)
	ed = dt.datetime(2011, 12, 31)

	#trade_df=[]
	
	#s_learner = sl.StrategyLearner(verbose = verbose, impact = impact)
	#t0 = time.clock()
	#s_learner.addEvidence(symbol = symbol, sd = sd, ed = ed, sv = sv)
	#t1 = time.clock()
	df= s_learner.testPolicy(symbol, sd, ed, sv)
	#df.columns = [str(i)]
	#t2 = time.clock()
	portvals = cp_sl(orders = df)
	portvals = pd.DataFrame(portvals)
	portvals.fillna(method = 'ffill', inplace=True)
	portvals.fillna(method = 'bfill', inplace = True)
	portvals=portvals/portvals.iloc[0]
	portvals.columns = [str(impact)]
	portvals_out = portvals
	trade_num_out = len(df)
	return portvals_in, trade_num_in, portvals_out, trade_num_out


if __name__=="__main__":

	portvals_in, trade_num_in, portvals_out, trade_num_out = experiment2(impact = 0.0)
	df_in= pd.DataFrame(portvals_in)
	count_in = []
	count_in.append(trade_num_in)
	df_out = pd.DataFrame(portvals_out)
	count_out = []
	count_out.append(trade_num_out)
	for i in [0.02,0.04,0.06,0.08]:
		df1, ct1, df2, ct2 = experiment2(impact = i)
		df1 = pd.DataFrame(df1)
		#print df
		count_in.append(ct1)
		df1.columns = [str(i)]
		df_in = df_in.join(df1)

		df2 = pd.DataFrame(df2)
		df2.columns = [str(i)]
		count_out.append(ct2)
		df_out = df_out.join(df2, how = 'right')
	# in sample
	print "count in sample: \n"
	print count_in
	print "count out sample: \n"
	print count_out
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	ax1 = df_in.plot(title = 'Experiment2-in sample')
	ax1.legend(loc='best')
	plt.savefig('Exp2_QLearner_{}_{}.png'.format(str(sd).split()[0], str(ed).split()[0]))
	plt.close('all')

	# out of sample
	
	sd = dt.datetime(2010, 1, 1)
	ed = dt.datetime(2011, 12, 31)
	ax2 = df_out.plot(title = 'Experiment2-out of sample') 
	ax2.legend(loc='best')
	plt.savefig('Exp2_QLearner_{}_{}.png'.format(str(sd).split()[0], str(ed).split()[0]))
	plt.close('all')
	
