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
import requests
import trade_api.db_api as db_api
collection_name = 'stockcandles'

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
        if type (prev_close) == pd.core.indexing._iLocIndexer and len(prev_close.obj) > 0:
            print(prev_close[0])
            df2["prev_c"] = prev_close[0]
            df2["chg"] = df2["close"] - df2["prev_c"]
    return df2

def calculate_spread(df2, prev_close=True):
    if "date" not in df2.columns:
        df2["date"] = df2.apply(lambda x: datetime.datetime(x.year, x.month, x.d_index), axis=1)
    if prev_close:
        df2 = calculate_chg(df2)
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
            m.write_df(df, collection_name)
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
    return df


def read_optionistics_stock(symbols, dirName):
    m = mongo_api()
    total_recs = 0
    df_out = None
    for (root,dirs,files) in os.walk(dirName, topdown=True):
        for fi in files:
            (file_type, d_cur, curr_symbol, fi) = parse_optionistics_filename(root, fi)
            print(file_type, fi)
            if file_type == 'OPTIONISTICS':
                print("reading ", fi)
                df = pd.read_csv(fi)
                df = persist_optionistics_stock_file(df, curr_symbol, m)
                df_out = append_df(df_out, df)
    total_recs = df_out.shape[0]
    print("total rec inserted", total_recs)
    df_out.to_pickle('today_stock.pickle')
    return total_recs


def update_close_for_df(symbols, df, prev_df):
    output_df = None
    for symbol in symbols:
        today_df = df[df["symbol"] == symbol]
        if today_df is None:
            continue
        prev_df_s = prev_df[prev_df["symbol"]==symbol]
        prev_close = prev_df_s["close"].iloc(0)
        today_df = calculate_chg(today_df, prev_close)
        output_df = append_df(output_df, today_df)
    return output_df


def update_multiple_date_close(df, dates, prev_df, symbols):
    symbols = list(np.unique(df["symbol"]))
    df_out = None
    for d in dates:
        print("setting for date", d)
        df_cur_date = df[df["date"] == d]
        assert(len(np.unique(prev_df.date)) == 1)
        assert(prev_df.date.iloc[0] < d)
        df_updated = update_close_for_df(symbols, df_cur_date, prev_df)
        #print("updated_df", df_updated)
        df_out = append_df(df_out, df_updated)
        prev_df = df_updated
    return df_out


def get_daily_stock_for_symbols(symbols, error_symbols, day_range):
    df_out = None
    symbol_count = 0
    d = Td.get_prev_trading_day(datetime.datetime.today())
    m = mongo_api()

    #assert(yesterday_frame.shape[0] == len(symbols))
    for symbol in symbols:
        try:
            #print("getting", symbol)
            df = getResultByType('price_history', '2048', symbol, "{\"resolution\": \"D\", \"count\":" + day_range + "}")
            if df is None:
                error_symbols.append(symbol)
                continue
            df["symbol"] = symbol
            df["date"] = df.t.apply(lambda x: datetime.datetime.fromtimestamp(np.int(x)) + datetime.timedelta(days=1))
            df = df.rename(columns = {"o": "open", "c": "close", "h": "high", "l": "low", "v": "volume"})
            d_index, month, year = df.date.apply(lambda x: x.day), \
                                   df.date.apply(lambda x: x.month), \
                                   df.date.apply(lambda x: x.year)
            df["d_index"] = d_index
            df["month"] = month
            df["year"] = year
            df = df.drop("s", axis=1)
            df = df.drop("t", axis=1)
            df = calculate_spread(df, False)
            #prev_close = yesterday_frame[yesterday_frame["symbol"] == symbol].iloc(0)
            #df = calculate_chg(df, prev_close)
            df_out = append_df(df_out, df)
            symbol_count += 1
            if symbol_count % 30 == 0:
                print("current processed ", symbol_count, " symbols")
                time.sleep(3)
        except requests.exceptions.SSLError:
            error_symbols.append(symbol)
            time.sleep(3)
            continue
        except json.decoder.JSONDecodeError:
            error_symbols.append(symbol)
            time.sleep(3)
            continue
    return df_out


def persist_daily_stock(day_range, symbol = None):

    #check if

    df_out = None
    if symbol is None:
        symbols = load_watch_lists("data/highvol_watchlist.json")
    else:
        symbols = [symbol]
    day_range = str(day_range)
    m = mongo_api()
    today = datetime.datetime.today()
    date_filter = db_api.construct_day_filter(today)
    if not db_api.check_persist_timing(m, 'optionstat', date_filter, today):
        return
    error_symbols = []
    symbols = np.sort(symbols)
    retryCount = 0

    while(1):
        df= get_daily_stock_for_symbols(symbols, error_symbols, day_range)
        df_out = append_df(df_out, df)
        retry_count = 0
        if (len(error_symbols) == 0 or df_out.shape[0] == len(symbols) or retry_count >= 3):
            break
        else:
            print("error symbols, continuing", error_symbols)
            retry_count = retry_count + 1

    if df_out is not None:
        print("saving...")
        today_str = datetime.datetime.today().strftime('%Y-%m-%d')
        yesterday = Td.get_prev_trading_day(datetime.datetime.today())
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        yesterday_frame = None
        if os.path.exists('stock.pickle_' + yesterday_str):
            yesterday_frame = pd.read_pickle('yesterday_stock.pickle')
        dates = list(np.unique(df_out["date"]))
        if yesterday_frame is not None:
            if len(dates) > 1:
                df_out["delete_flg"] = df_out.date.apply(lambda x: x.hour == 16)
                print("detect multiple dates", df_out.shape)
                df_out = df_out[df_out["delete_flg"] == False]
                dates = list(np.unique(df_out["date"]))
                df_out = df_out.drop("delete_flg", axis=1)
                print("drop out extra hour frame", df_out.shape)
                df_out = update_multiple_date_close(df_out, dates, yesterday_frame, symbols)
            else:
                df_out = update_close_for_df(symbols, df_out,yesterday_frame )
        #m.write_df(df_out, collection_name)
        df_out.to_pickle('stock.pickle' + today_str)
        print("total # of rows written", df_out.shape[0])


persist_daily_stock(4)
#df = pd.read_pickle('today_stock.pickle')
#dates = list(np.unique(df["date"]))
#print(dates)
#del dates[3]
#print(dates)
#today_ts = np.int((pd.Timestamp(dates[0]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
#today = datetime.datetime.fromtimestamp(today_ts)
#yesterday = Td.get_prev_trading_day(today)
#m = mongo_api()
#df_prev = m.read_df(collection_name, False, '*', [], {'$and': [{'symbol': {'$in': symbols}}, {'date': {'$eq': yesterday}}]}, {'symbol': 1, 'date': 1})
#output_df = update_multiple_date_close(df, dates, df_prev,symbols)
#output_df.to_pickle('updated.pickle')
#
#symbols = load_watch_lists("data/highvol_watchlist.json")
#print(len(symbols))
#read_optionistics_stock(symbols, "/home/jane/Downloads/data/200114/")
#symbols = np.sort(symbols)
#persist_stock_history_from_file(symbols)