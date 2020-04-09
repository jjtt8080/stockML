import json
import os as os

DEBUG_LEVEL = 0
import pandas as pd
import numpy as np
from trade_api.tda_api import Td
from trade_api.mongo_api import mongo_api

import tensorflow as tf
import random as rn
from keras import backend as K
import datetime
from optionML import getResultByType
import random
import time
import dateutil.relativedelta
from pathlib import Path

def set_random_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    # Setting the seed for numpy-generated random numbers
    np.random.seed(37)
    # Setting the seed for python random numbers
    rn.seed(1254)
    # Setting the graph-level random seed.
    if tf.__version__ == "2.0.0":
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        tf.set_random_seed(89)
        session_conf = tf.ConfigProto(
              intra_op_parallelism_threads=1,
              inter_op_parallelism_threads=1)

        #Force Tensorflow to use a single thread
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


def debug_print(*argv):
    if DEBUG_LEVEL==1:
        for arg in argv:
            print(arg)

def append_df(df_out, df):
    if df_out is None:
        df_out = df
    else:
        df_out = df_out.append(df, ignore_index=True)
    return df_out

def drop_columns(df, columnName):
    if type(columnName) == str and columnName in df.columns:
        return df.drop(columnName, axis=1)
    elif type(columnName) == list:
        for c in columnName:
            if c in df.columns:
                df = df.drop(c, axis=1)
    return df

def get_daily_symbols(watch_file_list):
    for f in watch_file_list:
        w = Td.load_json_file(f)


def load_watch_lists(watch_list_file):
    if not os.path.exists(watch_list_file):
        print ("file does not exist", watch_list_file)
        exit(2)

    watch_lists = load_json_file(watch_list_file)
    final_list = []
    for w in watch_lists["watch_lists"]:
        curr_watch = load_json_file(os.path.dirname(watch_list_file) + os.sep + w + ".json")
        final_list = Union(final_list, curr_watch[w])
    return final_list


def load_json_file(f):
    with open(f, 'r') as (wFile):
        data = json.load(wFile)
        return data


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def delta_days(d1, d2):
    return abs((d2 - d1).days)


def load_json_for_symbol(symbol):
    symbols = []
    if type(symbol) == str and symbol.find("'") != -1:
        symbol = symbol.replace("'", "\"")
    print(symbol)
    symbol = '{\"symbols\":' + symbol + '}'
    if symbol is not None:
        symbols = json.loads(symbol)
        return symbols["symbols"]
    return None


def calculate_chg(df2, prev_close=None):
    if df2 is None:
        return df2
    if prev_close is None:
        df2["prev_c"] = df2["close"].shift()
        i = list(df2.columns).index("prev_c")
        i2 = list(df2.columns).index("close")
        df2.iloc[0, i] = df2.iloc[0, i2]
    else:
        print(df2["symbol"])
        if type (prev_close) == pd.core.indexing._iLocIndexer and len(prev_close.obj)> 0:
            #print(prev_close[0])
            df2["prev_c"] = prev_close[0]
            if "close" not in df2.columns:
                print("can't find close in current df")
            else:
                df2["chg"] = df2["close"] - df2["prev_c"]
    return df2


def update_close_for_df(symbols, df, prev_df):
    output_df = None
    for symbol in symbols:
        today_df = df[df["symbol"] == symbol]
        if today_df is None or today_df.shape[0] == 0:
            continue
        prev_df_s = prev_df[prev_df["symbol"]==symbol]
        if prev_df_s is None or prev_df_s.shape[0] == 0:
            output_df = append_df(output_df, today_df)
            continue
        prev_close = prev_df_s["close"].iloc(0)
        today_df = calculate_chg(today_df, prev_close)
        output_df = append_df(output_df, today_df)
    if output_df is None:
        print("Can't find prev_dev or df", df, prev_df)
        output_df = df
    return output_df

def calculate_spread(df2, prev_close=True):
    if "date" not in df2.columns:
        df2["date"] = df2.apply(lambda x: datetime.datetime(x.year, x.month, x.d_index), axis=1)
    if prev_close:
        df2 = calculate_chg(df2)
    df2["ho_spread"] = df2["high"] - df2["open"]
    df2["cl_spread"] = df2["close"] - df2["low"]
    return df2

def post_process_ph_df(df, symbol):
    if "t" in df.columns and df.shape[0] > 0:
        if df[df.t.isna()].shape[0] > 0:
            print("wrong t", df.t, symbol)
        df["date"] = df.t.apply(lambda x: datetime.datetime.fromtimestamp(np.int(x)) + datetime.timedelta(days=1))
        df = drop_columns(df, ["s", "t"])
    if "c" in df.columns and "o" in df.columns:
        df = df.rename(columns={"o": "open", "c": "close", "h": "high", "l": "low", "v": "volume"})
    df["symbol"] = symbol
    d_index, month, year = df.date.apply(lambda x: x.day), \
                           df.date.apply(lambda x: x.month), \
                           df.date.apply(lambda x: x.year)
    df["d_index"] = d_index
    df["month"] = month
    df["year"] = year
    df = calculate_spread(df, False)
    return df

def calculate_td_parameter_from_finn_parameter(parameter):
    finn_parameter = json.loads(parameter)
    from_date = finn_parameter["from"]
    to_date = finn_parameter["to"]
    d1 = Td.convert_timestamp_to_time(from_date, "s")
    d1_date = datetime.date(d1.year, d1.month, d1.day)
    d2 = Td.convert_timestamp_to_time(to_date, "s")
    d2_date = datetime.date(d2.year, d2.month, d2.day)
    delta = dateutil.relativedelta.relativedelta(d1_date, d2_date)
    count = delta.months + 1
    td_parameter = {"periodType": "month", "period": count, "frequencyType": "daily", "frequency": 1}
    td_parameter_str = json.dumps(td_parameter)
    return td_parameter_str

def get_daily_stock_for_intraday(symbol, input_date):
    df = getResultByType('price_history', None, symbol, "{\"resolution\": \"60\", \"count\": \"200\"  }")
    df["date"] = df.t.apply(lambda x: Td.convert_timestamp_to_time(x, 's'))
    df["keep_flg"] = df["date"].apply(lambda x:  (x.hour >= 6) & (x.hour <=16) & (x.year == input_date.year) & (x.month == input_date.month) & (x.day == input_date.day))
    df = df[df["keep_flg"] == True]
    df = df.sort_values("date")
    df["volume"] = np.sum(df.v)
    df["high"] = np.max(df.h)
    df["low"] = np.min(df.l)
    df["open"] = df["o"].iloc[0]
    df["close"] = df["c"].iloc[-1]
    df = drop_columns(df, ["o","c","v", "h", "l", "s", "t"])
    df["date"] = df.date.apply(lambda x: x.replace(hour=21))
    df["symbol"] = symbol
    return df.head(1)

def get_single_stockcandles(symbol, parameter, df_out, useIntraDay = False, input_day = None):
    df_symbol = None
    if useIntraDay:
        df_symbol = get_daily_stock_for_intraday(symbol,input_day)
    else:
        df_symbol = getResultByType('price_history', None, symbol, parameter)
    if df_symbol is not None:
        df_symbol["symbol"] = symbol
        df_symbol = post_process_ph_df(df_symbol, symbol)
        df_out = append_df(df_out, df_symbol)
    else:
        print("get_single_stockcandles failed to get history", symbol, parameter, df_out)
    return df_out


def get_stockcandles_for_day(d1, d2, watch_list_file_name=None, symbols=None, useIntraDay=False):
    d_str_1 = datetime.datetime.strptime(d1, "%Y%m%d")
    if useIntraDay:
        d_str_1 = d_str_1.replace(hour=6,minute=30)
    d_str_2 = datetime.datetime.strptime(d2, "%Y%m%d")
    if useIntraDay:
        d_str_2 = d_str_2.replace(hour=21,minute=0)
    d1_t = d_str_1.strftime("%s")
    d2_t = d_str_2.strftime("%s")
    if useIntraDay:
        parameter = "{\"resolution\": \"D\", \"from\": \"" + d1_t + "\",\"to\": " + "\"" + d2_t + "\"}"
    else:
        deltaDays = delta_days(d_str_1, d_str_2)
        parameter = "{\"resolution\": \"D\",  \"from\": \"" + d1_t + "\",\"count\": " + str(deltaDays) + "}"
    if watch_list_file_name is not None:
        symbols = load_watch_lists(watch_list_file_name)

    df_out = None
    error_symbols = set()
    done_symbols = set()
    for symbol in symbols:
        try:
            print("getting ", symbol)
            df_out = get_single_stockcandles(symbol, parameter, df_out, useIntraDay, d_str_2)

            if df_out is not None and df_out.shape[0] > 0:
                done_symbols.add(symbol)
            else:
                error_symbols.add(symbol)
            if len(done_symbols) % 30 == 0:
                time.sleep(3)
        except json.decoder.JSONDecodeError:
            error_symbols.add(symbol)
            continue
        except KeyError:
            continue


    retry = 0
    while len(error_symbols) > 0 and retry <= 3:
        for symbol in error_symbols:
            try:
                df_out = get_single_stockcandles(symbol, parameter, df_out, useIntraDay, d_str_2)
                if df_out is not None:
                    df_out = post_process_ph_df(df_out, symbol)
                    done_symbols.add(symbol)
            except json.decoder.JSONDecodeError:
                continue
        error_remaining = set(symbols) - done_symbols
        error_symbols = error_remaining
        retry += 1
    print("error on symbols", error_symbols)
    temp = str(random.randint(0,50))
    file_suffix = d_str_2.strftime("%Y-%m-%d")
    filename = "stock.pickle" + file_suffix + "_" + temp
    print("persist to file at get_stockcandles_for_day ", filename)
    if df_out is not None:
        df_out.to_pickle(filename)
    else:
        print("nothing persisted")
    return df_out


def post_processing_for_close(df, d):
    prev_trading_date = Td.get_prev_trading_day(d)
    print("prev_trading_date", prev_trading_date)
    m = mongo_api()
    df_prev= m.read_df('stockcandles', False, '*', [],\
                                     {'$and': [{'year': {'$eq': prev_trading_date.year}},\
                                               {'month':{'$eq': prev_trading_date.month}},\
                                               {'d_index': {'$eq': prev_trading_date.day}}]},{})
    print(df_prev.columns)
    symbols = set(df_prev["symbol"])
    output_df = update_close_for_df(symbols, df, df_prev)
    output_df["date"] = output_df.date.apply(lambda x: datetime.datetime(x.year, x.month, x.day, 0,0,0))
    return output_df


def update_close_forall(date_range):
    m = mongo_api()
    d1,d2 = date_range.split(",")
    date1 = datetime.datetime.strptime(d1, '%Y%m%d');
    date2 = datetime.datetime.strptime(d2, '%Y%m%d')
    df_all =  m.read_df('stockcandles', False, '*', [], {'$and': [{'date': {'$gte': date1}}, {'date':{'$lte': date2}}]}, {})
    df_all = df_all.sort_values(["symbol", "date"])
    df_out = None
    for s in set(df_all.symbol):
        df_s = df_all.loc[df_all.symbol == s]
        df_s = df_s.sort_values("date")
        df_s_first_index = df_s.index[0]
        df_s["prev_c"] = df_s["close"].shift(periods=1, fill_value=0)
        df_s.loc[df_s_first_index, "prev_c"] = df_s.loc[df_s_first_index, "close"]
        df_s["chg"] = df_s["close"] - df_s["prev_c"]
        df_out = append_df(df_out, df_s)
    return df_out

def get_home_dir():
    home = str(Path.home())
    return home