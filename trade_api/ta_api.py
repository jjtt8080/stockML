import ta
import datetime
import sys
import os
import getopt
import pandas as pd
import numpy as np

sys.path.append("..")
from trade_api.mongo_api import mongo_api
from trade_api.tda_api import Td
from scipy.signal import argrelextrema
from util import append_df

def get_features(df, options=None):
    if options == 'all':
        df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume")
    elif options == 'volume':
        df = ta.add_volume_ta(df, "high", "low", "close", "volume")
    elif options == 'momentum':
        df = ta.add_momentum_ta(df, "high", "low", "close", "volume")
    elif options == 'volatility':
        df = ta.add_volatility_ta(df, "high", "low", "close")
    elif options == 'trend':
        df = ta.add_trend_ta(df, "high", "low", "close")
    elif options == 'others':
        df = ta.add_others_ta(df, "close")
    df = df.sort_values('date')
    return df


def get_max_min(prices, smoothing, window_range):
    # assuming input is already sorted by date column and index is sequentially ascending
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range]['close'].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    #max_min.index.name = 'timestamp'
    #max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    p = prices
    max_min['day_num'] = p[p['date'].isin(max_min.date)].index.values
    max_min = max_min.set_index('day_num')['close']
    return max_min


def get_vwap(df):
    total_volume = np.sum(df["volume"])
    df["vwap"] = df.apply(lambda x:( x["close"] * x["volume"]) /  total_volume, axis=1)
    return df

def get_swingindex(df):
    m = mongo_api()
    df_prevFrame = None
    i = 0
    symbolSet = set(df.symbol)
    df_out = None
    for s in symbolSet:
        df_symbol = df[df.symbol == s]
        df_symbol = df_symbol.sort_values("date")
        df_symbol = df_symbol.reset_index()
        prev_trading_date = Td.get_prev_trading_day(df.iloc[0].date)
        df_prevFrameSymbol = None
        if df_prevFrame is None:
            df_prevFrame = m.read_df('stockcandles', False, '*', [], { 'date': {'$eq': prev_trading_date}}, {})
            print("read prev day", df_prevFrame.shape)
        df_prevFrameSymbol = df_prevFrame[df_prevFrame.symbol == s]
        if df_prevFrameSymbol is None or df_prevFrameSymbol.shape[0] == 0:
            print("can't find prev day frame:", s)
            continue
        #print("prev_day symbol", s, df_prevFrameSymbol.shape)
        df_symbol["today_chg"] = df_symbol.apply(lambda x: x.close - x.open, axis=1)
        df_symbol["prev_chg"] = df_symbol.today_chg.shift()
        min_date = np.min(df_symbol.date)
        df_symbol.loc[df_symbol.date == min_date, "prev_chg"] = df_prevFrameSymbol["close"].iloc[0] - df_prevFrameSymbol["open"].iloc[0]
        df_symbol["weighted_change"] = df_symbol.apply(lambda x: 50 * (x.chg + 0.5 * x.prev_chg + 0.25 * x.today_chg), axis=1)
        df_symbol["R1"] = df_symbol.apply(lambda x: np.abs(x.high - x.prev_c) - 0.5 * np.abs(x.low - x.prev_c)  + 0.25 * x.prev_chg, axis=1)
        df_symbol["R2"] = df_symbol.apply(lambda x: np.abs(x.low - x.prev_c) - 0.5 * np.abs(x.high - x.prev_c) + 0.25 * x.prev_chg, axis=1)
        df_symbol["R3"] = df_symbol.apply(lambda x: np.abs(x.high - x.low) + 0.25 * x.prev_chg, axis=1)
        df_symbol["R"] = df_symbol.apply(lambda x: np.amax([x.R1, x.R2, x.R3]), axis=1)
        df_K = np.max(df_symbol.apply(lambda x: np.amax([x.high - x.prev_c, x.low - x.prev_c]), axis=1))
        df_M = np.max(df_symbol.apply(lambda x: (x.high - x.low), axis=1))
        df_symbol["si"] = df_symbol.apply(lambda x: (x["weighted_change"] / x.R) * (df_K / df_M), axis=1)
        i += 1
        if i % 20 == 0:
            print("processed ", i)
        df_out = append_df(df_out, df_symbol)
    return df_out


#smoothing = 3
#window = 10
#minmax = get_max_min(resampled_data, smoothing, window)
#minmax

def read_data(symbol, date_begin, date_end):
    m = mongo_api()
    date_range = Td.get_market_hour(date_begin, date_end)
    print(date_range["market_day"][0], date_range["market_day"][-1])
    df = m.read_df('stockcandles', False, '*', [], {'symbol': {'$eq': symbol}}, {'date':1})
    df = df[(df.date >= date_range["market_day"][0]) & (df.date <= date_range["market_day"][-1])]
    print(df.shape)
    return df


def main(argv):
    symbol = None
    watch_list_file = None
    date_begin = None;
    date_end = None
    ta_options = None
    try:
        opts, args = getopt.getopt(argv, "hd:s:w:t:",
                                   ["help", "d=", "s=", "w=", "t="])
        for opt, arg in opts:
            if opt == '-d':
                date_begin, date_end = arg.split(',')
                print(date_begin, date_end)
            elif opt == '-s':
                symbol = arg
            elif opt == '-w':
                watch_list_file = arg
            elif opt == '-h':
                print("usage: -d <date_begin,date_end> -s symbol -w watch_list_file")
                exit(1)
            elif opt == '-t':
                ta_options = arg

    except getopt.error as e:
        print("argument error", e)
        exit(-1)
    print(symbol)
    if symbol is not None:
        print("getting technical analysis for symbol", symbol)
        df = read_data(symbol, date_begin, date_end)
        df = get_features(df, ta_options)
        print(df.columns)
        print(df)


if __name__ == "__main__":
    main(sys.argv[1:])
