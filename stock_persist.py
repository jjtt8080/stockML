import datetime
import json
import sys
import time

import numpy as np
import pandas as pd

from trade_api.mongo_api import mongo_api

sys.path.append(".")
from optionML import getResultByType
from trade_api.tda_api import Td
from trade_api.db_api import construct_day_filter
from util import debug_print, append_df, update_close_for_df, load_watch_lists, drop_columns, calculate_spread,post_process_ph_df, get_daily_stock_for_intraday
import os
import requests
import trade_api.db_api as db_api
collection_name = 'stockcandles'
pickles_dir = 'stock_pickles'
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
                    print("can't get result for symbol", symbol)
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


def get_daily_stock_from_td(symbol):
        df_td = getResultByType('price_history', '2048', symbol,
                                "{\"periodType\": \"month\",\"frequencyType\": \"daily\", \"period\":1, \"frequency\": 1}")
        if df_td is not None:
            df_td = df_td[df_td.index == max(df_td.index)]
            df = df_td.rename(columns={"datetime": "date"})

            return df
        else:
            return None




def get_daily_stock_for_symbols(symbols, error_symbols, done_symbols, day_range, use_td=False, use_intraday=True):
    df_out = None
    symbol_count = 0
    d = Td.get_prev_trading_day(datetime.datetime.today())
    m = mongo_api()

    #assert(yesterday_frame.shape[0] == len(symbols))
    for symbol in symbols:
        if symbol in done_symbols:
            continue
        try:
            print("getting", symbol)
            #first we use the finn API
            if use_td:
                df = get_daily_stock_from_td(symbol)
            elif use_intraday:
                df = get_daily_stock_for_intraday(symbol, datetime.datetime.today())
            else:
                df = getResultByType('price_history', None, symbol, "{\"resolution\": \"D\", \"count\":" + day_range + "}")
            if df is None and use_td == False:
                df = get_daily_stock_from_td(symbol)
            else:
                df = post_process_ph_df(df, symbol)
            if df is None:
                print("add to error symbols", symbol)
                error_symbols.add(symbol)
                continue

            #prev_close = yesterday_frame[yesterday_frame["symbol"] == symbol].iloc(0)
            #df = calculate_chg(df, prev_close)
            df_out = append_df(df_out, df)
            symbol_count += 1
            if symbol in error_symbols:
                error_symbols.remove(symbol)
            done_symbols.add(symbol)
            if symbol_count % 15 == 0:
                print("current processed ", symbol_count, " symbols")
                time.sleep(3)
        except requests.exceptions.SSLError:
            error_symbols.add(symbol)
            if len(error_symbols) % 15 == 0:
                print("error symbols:", error_symbols)
            time.sleep(3)
            continue
        except json.decoder.JSONDecodeError:
            error_symbols.add(symbol)
            if len(error_symbols) % 15 == 0:
                print("error symbols:", error_symbols)
            time.sleep(3)
            continue
    return df_out


def compose_file_name(d):
    day_str = d.strftime('%Y-%m-%d')
    filename = pickles_dir + os.sep +  'stock.pickle' + day_str
    if os.path.exists(filename):
        day_str = d.strftime('%Y-%m-%d-%h:%M:%s')
        filename = 'stock.pickle' + day_str
    return filename


def post_processing(df_all, symbols, error_symbols):
    if df_all is not None and df_all.shape[0] > 0:
        print("saving...", df_all.shape)
        #today_str = datetime.datetime.today().strftime('%Y-%m-%d')
        dates = list(set(df_all["date"]))
        first_day_str = min(dates).strftime('%Y-%m-%d')
        yesterday = Td.get_prev_trading_day(min(dates))
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        yesterday_frame = None
        print("yesterday_str", yesterday_str)
        if os.path.exists( pickles_dir + os.sep + 'stock.pickle' + yesterday_str):
            yesterday_frame = pd.read_pickle( pickles_dir + os.sep + 'stock.pickle' + yesterday_str)
            print("yesterday's frame", yesterday_frame.shape, yesterday_frame.columns)
        else:
            m = mongo_api()
            date_filter = construct_day_filter(yesterday)
            yesterday_frame = m.read_df('stockcandles', False, '*', [], date_filter, {})
            print("yesterday's frame", yesterday_frame.shape, yesterday_frame.columns)

        if yesterday_frame is not None and yesterday_frame.shape[0] > 0:
            if len(dates) > 1:
                df_all["delete_flg"] = df_all.date.apply(lambda x: x.hour == 16)
                print("detect multiple dates", df_all.shape)
                df_all = df_all[df_all["delete_flg"] == False]
                dates = list(np.unique(df_all["date"]))

                df_all = update_multiple_date_close(df_all, dates, yesterday_frame, symbols)
            else:
                df_all = update_close_for_df(symbols, df_all, yesterday_frame)
        #m.write_df(df_out, collection_name)

        print(df_all.shape, set(df_all.date))
        df_all.to_pickle('stock_pickles/stock.pickle_tmp' + "_"+ datetime.datetime.today().strftime("%Y%m%d%H%M%S"))
        df_all["delete_flag"] = df_all.date.apply(lambda x: (x.hour == 16))
        df_all["keep_flag"] = df_all.date.apply(lambda x: (x.hour == 21))

        df_keep = df_all[df_all["keep_flag"] == True]
        print("df_keep.shape", df_keep.shape)
        df_discard = df_all[df_all["delete_flag"] == True]
        print("df_discard.shape", df_discard.shape)
        df_keep = drop_columns(df_keep, ["delete_flag", "keep_fJun19 85 C lag"])
        df_discard = drop_columns(df_discard,  ["delete_flag", "keep_flag"])
        m = mongo_api()
        if df_keep.shape[0] > 0:
            df_keep["date"] = df_keep.date.apply(lambda x: datetime.datetime(x.year, x.month, x.day, 0,0,0))
            for d in set(df_keep.date):
                filename = compose_file_name(d)
                df_keep[df_keep["date"] == d].to_pickle(filename)
                m.write_df(df_keep, collection_name)
                print("total # of rows written, error symbols", df_keep.shape[0], len(error_symbols))
        if df_discard.shape[0] > 0:
            for d in set(df_discard.date):
                filename = compose_file_name(d) + "_discard"
                df_discard[df_discard["date"] == d].to_pickle(filename)
                print("total # of rows written for timestamp 16:00:00, error symbols", df_discard.shape[0], len(error_symbols))


def persist_daily_stock(day_range, watch_list_file, symbol = None, useIntraDay=False):
    skipped_set = {"VIAB", "CBS", "BBT"}
    df_all = None
    if symbol is None:
        symbols = load_watch_lists(watch_list_file)
    else:
        symbols = [symbol]
    day_range = str(day_range)
    m = mongo_api()
    today = datetime.datetime.today()
    date_filter = db_api.construct_day_filter(today)
    projection = "date"
    #if not db_api.check_persist_timing(m, 'optionstat', projection, date_filter, today):
    #    return
    error_symbols = set()
    symbols = np.sort(symbols)
    retryCount = 0
    symbols_set = set(symbols)
    for skip_s in skipped_set:
        if skip_s in symbols_set:
            symbols_set.remove(skip_s)
    symbols = list(symbols_set)
    done_symbols = set()
    retry_count = 0
    while(len(done_symbols) != len(symbols)):
        df= get_daily_stock_for_symbols(symbols, error_symbols, done_symbols, day_range, False, useIntraDay)
        df_all = append_df(df_all, df)
        if (len(error_symbols) == 0 or (df_all is not None and df_all.shape[0] == len(symbols)) or retry_count >= 4):
            post_processing(df_all,symbols, error_symbols)
            if len(set(df_all.symbol)) + len(skipped_set) < len(symbols):
                print("missing", set(symbols) - set(df_all.symbols) - set(skipped_set))
            break
        else:
            #post_processing(df_all, symbols, error_symbols)
            print("error symbols, continuing, resetting symbols to error_Symbols", error_symbols)
            retry_count = retry_count + 1


if sys.argv[1] is None or sys.argv[1] == '':
    print("Error argument, pass in a watch list name")
    exit(1)
if len(sys.argv) >= 3 and sys.argv[2] is not None:
    day_range = sys.argv[2]
else:
    day_range = 1
symbol = None
useIntraDay = True
optionistic = False
if len(sys.argv) >=4 and sys.argv[3] != "":
    symbol = sys.argv[3]
if len(sys.argv) >= 5 and sys.argv[4] != "":
    useIntraDay = False 
if len(sys.argv) >= 6 and sys.argv[5] != "":
    optionistic = True
if optionistic is False:
    persist_daily_stock(day_range, sys.argv[1], symbol, useIntraDay)
#else:
#rec = read_optionistics_stock([sys.argv[1]], "/home/jane/Downloads/20200403")
#print("totally ", rec , "records")
