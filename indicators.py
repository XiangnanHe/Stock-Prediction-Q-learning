#Name: Xiangnan He
#ID: xhe321

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#implement two of SMA, Bollinger Bands, RSI, and implement another outside of these three, I chose Momentum

def bol_bands(df, symbol, window = 20):
    #bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
    roll_mean = pd.rolling_mean(df[symbol], window = window)
    roll_std = pd.rolling_std(df[symbol], window = window)
    upper_band = roll_mean+2*roll_std
    lower_band = roll_mean-2*roll_std
    bb_value = (df[symbol] - roll_mean)/(2*roll_std)
    return upper_band, lower_band, roll_mean, bb_value



def SMA(df, symbol, window = 20):
    price = df[symbol]
    price = price/price[window - 1]
    roll_mean = pd.rolling_mean(df[symbol], window = window)
    #roll_mean = roll_mean/roll_mean[window - 1] #normalize
    div = df[symbol]/roll_mean
    #div = div/div[window - 1] # normalize
    return (price,roll_mean,div)
    
    

def momentum(df, symbol, window = 10):
    #momentum[t] = (price[t]/price[t-N]) - 1
    momentum = df[symbol]
    momentum = momentum/momentum.shift(window) - 1
    #normalize
    #momentum = momentum/momentum[window]
    price = df[symbol]
    #price = price/price[0]

    return price,momentum
    

