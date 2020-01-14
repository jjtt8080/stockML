import datetime
import time
import sys
import numpy as np
import pandas as pd
import json
from trade_api.mongo_api import mongo_api
sys.path.append(".")
from optionML import getResultByType
from trade_api.tda_api import Td
from util import debug_print, append_df, load_watch_lists
import os
import trade_api.db_api as db_api


STOCK_HIST_FILE_PATH = 'data/historical//pickles/stockhist'

def get_market_days():
    start_date = '2015-01-02'
    end_date = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')
    market_hours = Td.get_market_hour(start_date, end_date)
    dest_data = set(market_hours["market_day"])
    return (dest_data, start_date, end_date)


def persist_stock_price_history(symbols):
        m = mongo_api()
        (dest_data, start_date, end_date) = get_market_days()
        total_recs = 0

        for symbol in symbols:
            try:
                db_stock_df = m.read_df('stockhist', True, "datetime", [], {'symbol': {'$eq': symbol}} ,{"datetime":1})
                if db_stock_df is not None and db_stock_df.shape[0] > 0 and "datetime" in db_stock_df.columns:
                    db_stock_df["market_day"] = db_stock_df["datetime"].apply(lambda x: datetime.datetime(x.year, x.month, x.day, 0,0,0))
                    curr_data = set(db_stock_df["market_day"])
                    diff_date = np.array(list(dest_data - curr_data))
                else:
                    diff_date = np.array(list(dest_data))
                diff_date = np.sort(diff_date)
                #debug_print("Differentiated dates", len(diff_date))
                if len(diff_date) <=0:
                    continue
                m.deleteMany('stockhist', {'symbol': {'$eq': symbol}})
                start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                delta = (datetime.datetime.today() - start_datetime).days + 1
                option_params = "{\"resolution\" : \"D\", \"count\": " + str(delta) + "}"
                df = getResultByType('price_history', '2048', symbol, option_params)
                if df is None:
                    continue
                df["datetime"] = df.t.apply(lambda x: Td.convert_timestamp_to_time(x, 's'))
                df["symbol"] = symbol
                #df = df.sort_values(['datetime'])
                # make sure we get the same shape
                df = df.sort_values('datetime',ascending=True)
                market_day = df.datetime.apply(
                    lambda x: datetime.datetime(x.year, x.month, x.day, 0, 0, 0))
                if (len(set(market_day)) < len(dest_data)):
                    print("length diff", symbol, len(market_day), len(dest_data))
                debug_print("read stock history", df.shape)
                diff_ts = []
                for d in diff_date:
                    diff_ts.append((np.datetime64(d) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
                df["ts"] = df.datetime.apply(lambda x: (np.datetime64(x.strftime('%Y-%m-%dT00:00:00Z')) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
                debug_print("df.ts", df.ts)
                debug_print("diff_ts", diff_ts)
                df = df[df["ts"].isin(diff_ts)]
                debug_print("df.shape after filter", df.shape)
                if df.shape[0] > 0:
                    m.write_df(df, 'stockhist')
                    total_recs += df.shape[0]
                else:
                    total_recs += 0
            except KeyError:
                print("error when persist stock price history")
                continue
                return 0
        return total_recs


def calculate_spread(df2):
    if "date" not in df2.columns:
        df2["date"] = df2.apply(lambda x: datetime.datetime(x.year, x.month, x.d_index), axis=1)
    df2["prev_c"] = df2["close"].shift()
    i = list(df2.columns).index("prev_c")
    i2 = list(df2.columns).index("close")
    df2.iloc[0, i] = df2.iloc[0, i2]
    df2["chg"] = df2["close"] - df2["prev_c"]
    df2["ho_spread"] = df2["high"] - df2["open"]
    df2["cl_spread"] = df2["close"] - df2["low"]
    return df2


def persist_stock_history_from_file(symbols):
    # we have two files which has different columns
    m = mongo_api()
    total_recs = 0
    for year in 2015,2016,2017, 2018, 2019:
        df = pd.read_pickle(STOCK_HIST_FILE_PATH + os.sep + str(year) + '.pickle_stock')
        df = df[df["symbol"].isin(symbols)]
        try:
            df = calculate_spread(df)
            print(df.shape, df.columns)
            total_recs += df.shape[0]
            m.write_df(df, "stockcandles")
        except KeyError:
            print("error when persist stock price history")
            continue
            return 0
    print("total records inserted", total_recs)
    return total_recs


def persist_company_info(symbol):
    try:
        if db_api.recordExists('stock_quotes', {'symbol': {'$eq': symbol}}):
            print("skipped symbol", symbol)
            return
        quotes = getResultByType('quote', '2048', symbol, {}, True)
        df = pd.DataFrame(data= quotes)
        df = df.transpose()
        df = df.reset_index()
        df = df.rename(columns = {"index": "symbol"})
        m = mongo_api()
        m.write_df(df, 'stock_quotes')
    except ValueError:
        print("error persist ", symbol)


def getStockDesc(symbols, descName):
    m = mongo_api()
    df = m.read_df('stock_quotes', True, ['symbol', descName], [], {"symbol": {'$in': symbols}}, {"symbol":1})
    assert(df.shape[0] == len(symbols))
    return df[descName]


def parse_optionistics_filename(root, fi):
    original_f = fi
    tokens = fi.split(".")
    if len(tokens) != 5 or tokens[1] != "stock" or tokens[4] != "csv":
        return ('Unknown', None, None, fi)
    curr_symbol = tokens[0].upper()
    d_cur = datetime.datetime.strptime(tokens[2], '%Y%m%d')
    fi = root + os.sep + fi
    return ('OPTIONISTICS', d_cur, curr_symbol, fi)


def lstrip_s(s):
    return s.lstrip()


def persist_optionistics_stock_file(df, symbol, m):
    num_records_inserted = 0
    # debug_print(df.columns)
    columns = df.columns
    new_columns = map(lstrip_s, columns)
    df = df.rename(str.lstrip, axis="columns")
    df["close"] = df["last"]
    df = df.drop("option volume", axis=1)

    df["symbol"] = symbol
    df["date"] = df.date.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    d_index, month, year = df.date.apply(lambda x: x.day),\
                              df.date.apply(lambda x: x.month), \
                              df.date.apply(lambda x: x.year)
    df["d_index"] = d_index
    df["month"] = month
    df["year"] = year
    df = calculate_spread(df)
    m.write_df(df, 'stockcandles')
    return df.shape[0]


def read_optionistics_stock(symbols, dirName):
    m = mongo_api()
    total_recs = 0
    for (root,dirs,files) in os.walk(dirName, topdown=True):
        for fi in files:
            (file_type, d_cur, curr_symbol, fi) = parse_optionistics_filename(root, fi)
            if file_type == 'OPTIONISTICS':
                print("reading ", fi)
                df = pd.read_csv(fi)
                total_recs += persist_optionistics_stock_file(df, curr_symbol, m)
    print("total rec inserted" , total_recs)
    return total_recs


def persist_daily_stock_for_symbols(symbols, error_symbols, day_range):
    df_out = None
    symbol_count = 0
    for symbol in symbols:

        try:
            #print("getting", symbol)
            df = getResultByType('price_history', '2048', symbol, "{\"resolution\": \"D\", \"count\":" + day_range + "}")
            if df is None:
                error_symbols.append(symbol)
                continue
            df["symbol"] = symbol
            df["date"] = df.t.apply(lambda x: datetime.datetime.fromtimestamp(np.int(x)))
            df = df.rename(columns = {"o": "open", "c": "close", "h": "high", "l": "low", "v": "volume"})
            d_index, month, year = df.date.apply(lambda x: x.day), \
                                   df.date.apply(lambda x: x.month), \
                                   df.date.apply(lambda x: x.year)
            df["d_index"] = d_index
            df["month"] = month
            df["year"] = year
            df = calculate_spread(df)
            df_out = append_df(df_out, df)
            symbol_count += 1
            if symbol_count % 30 == 0:
                time.sleep(3)
        except json.decoder.JSONDecodeError:
            time.sleep(3)
            continue
    if df_out is not None:
        if not os.path.exists('today_stock.pickle'):
            df_out.to_pickle('today_stock.pickle')
        else:
            df = pd.read_pickle('today_stock.pickle')
            df_out = append_df(df_out, df)
            df_out.to_pickle('today_stock.pickle')
    return df_out


def persist_daily_stock(day_range):
    df_out = None
    symbols = load_watch_lists("data/watch_list.json")
    day_range = str(day_range)
    m = mongo_api()
    error_symbols = []
    symbols = np.sort(symbols)
    retryCount = 0
    while(1):
        df= persist_daily_stock_for_symbols(symbols, error_symbols, day_range)
        df_out = append_df(df_out, df)
        if len(error_symbols) or df_out.shape[0]:
            break
        else:
            print("error symbols, continuing", error_symbols)
    if df_out is not None:
        m.write_df(df_out, 'stockcandles')
    print("total # of rows written", df_out.shape[0])

persist_daily_stock(2)
#symbols = Td.load_json_file("data/high_vol.json")["high_vol"]

#read_optionistics_stock(symbols, "/home/jane/Downloads/stocks")
#symbols = np.sort(symbols)
#persist_stock_history_from_file(symbols)