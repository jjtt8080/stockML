# This file persist the option detail data we obtained from optiondata website
# to the optionhist collection in mongo
#  it also persist the data we daily grab from tda
import pandas as pd
import numpy as np
import os as os
import sys as sys
import json
from trade_api.mongo_api import mongo_api
import getopt
import csv
import datetime
from analyze_historical import OptionStats
from optionML import getResultByType
from trade_api.tda_api import Td
from analyze_historical import OptionStats
import timedelta
<<<<<<< HEAD
from util import debug_print, append_df, load_watch_lists
from trade_api.db_api import read_one_stock_quote, getStockDesc, get_stock_price_history, recordExists
from trade_api.calculate_iv import newton_vol_call,newton_vol_put,newton_vol_call_div,newton_vol_put_div
=======
DEBUG_LEVEL = 1


def debug_print(*argv):
    if DEBUG_LEVEL==1:
        for arg in argv:
            print(arg)


def persist_stock_price_history(symbol):
    try:
        m = mongo_api()
        start_date = '2015-12-02'
        end_date = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')
        market_hours = Td.get_market_hour(start_date, end_date)
        #First check what we have in the mongo DB
        dest_data = set(market_hours["market_day"])
        db_stock_df = m.read_df('stockhist', True, "datetime", [], {} ,{"datetime":1})
        if db_stock_df is not None and db_stock_df.shape[0] > 0:
            db_stock_df["market_day"] = db_stock_df.iloc[:,0].apply(lambda x: datetime.datetime(x.year, x.month, x.day, 0,0,0))
            curr_data = set(db_stock_df["market_day"])
            diff_date = np.array(list(dest_data - curr_data))
        else:
            diff_date = np.array(list(dest_data))
        diff_date = np.sort(diff_date)
        print("Differentiated dates", len(diff_date))
        if len(diff_date) <=0:
            return
        delta = (diff_date[len(diff_date)-1] - diff_date[0]).days + 1
        option_params = "{\"resolution\" : \"D\", \"count\": " + str(delta) + "}"
        df = getResultByType('price_history', '2048', symbol, option_params)
        df["datetime"] = df.t.apply(lambda x: Td.convert_timestamp_to_time(x, 's'))
        df["symbol"] = symbol
        df = df.sort_values(['datetime'])
        debug_print("read stock history", df.shape)
        debug_print(df.datetime)
        m.write_df(df, 'stockhist')
        return df.shape[0]
    except KeyError:
        print("error when persist stock price history")
        return 0


def read_single_date_price(symbol,currDate):
    m = mongo_api()
    if currDate.isoweekday() == 6:
        currDate = currDate - datetime.timedelta(1)
    if currDate.isoweekday() == 7:
        currDate = currDate - datetime.timedelta(2)
    debug_print("1.1, currDate", currDate,  currDate + datetime.timedelta(1))
    debug_print("1.2", currDate + datetime.timedelta(1))
    date_filter = {'datetime': {'$gt': currDate, '$lt': currDate + datetime.timedelta(1)}}
    try:
        db_df = m.read_df('stockhist', False, ['c', 'datetime'], [], {'$and': [{'symbol': {'$eq': symbol}}, date_filter]}, None)
        if db_df.shape is not None and db_df.shape[0] >= 1:
            debug_print("found price for ", currDate, " at ", db_df.iloc[0,1])
            return db_df.iloc[0, 1]
        else:
            print("can't find specific symbol")
            return None
    except KeyError:
        print("error when reading single stock price history")
        return None


def get_stock_price_history(symbol, currDate):
    try:
        p = read_single_date_price(symbol,currDate)
        if p is None:
            debug_print("can't find the symbol in specific date", symbol, currDate)
        return p
    except TypeError:
        print("error when get stock price history")
        exit(1)
>>>>>>> bbba58a92acd0e075c0b271c3f54bb4d4623e4ed

DEBUG_LEVEL = 1

def construct_option_symbol(x):
    dateStr = x.exp_date.strftime("%Y%m%d")[2:]
    retVal = x.UnderlyingSymbol + dateStr
    if x.Type == 'call':
        retVal += 'C'
    else:
        retVal += 'P'
    strikeStr = str(int(x.Strike * 1000))
    if len(strikeStr) < 8:
        strikeStr = strikeStr.zfill(8)
    retVal += strikeStr
    return retVal


def parse_optionistics_filename(root, fi):
    original_f = fi
    tokens = fi.split(".")
    if len(tokens) != 5 or tokens[1] != "options" or tokens[4] != "csv":
        return ('Unknown', None, None, fi)
    curr_symbol = tokens[0].upper()
    d_cur = datetime.datetime.strptime(tokens[2], '%Y%m%d')
    fi = root + os.sep + fi
    return ('OPTIONISTICS', d_cur, curr_symbol, fi)


def determine_file_origin(root, fi):
    original_f = fi
    debug_print("determine file origin", fi)
    if original_f[7:13] != 'option':
        return parse_optionistics_filename(root, fi)
    fi = root + os.sep + fi
    #debug_print(fi)
    parent_dirs = fi.split("/")
    data_date = "20" + original_f[0:6]
    d_cur = datetime.datetime.strptime(data_date, "%Y%m%d")
    curr_symbol = parent_dirs[-2]
    return ('TDA', d_cur, curr_symbol, fi)


def derive_columns(df, d_cur):
    df["data_date"] = d_cur
    df["days_to_expire"] = df["exp_date"].apply(lambda x: (x - d_cur).days)
    df["exp_year"] = df["exp_date"].apply(lambda x: x.year)
    df["exp_month"] = df["exp_date"].apply(lambda x: x.month)
    df["exp_week"] = df["exp_date"].apply(lambda x: OptionStats.week_number_of_month(x))
    df["exp_day"] = df["exp_date"].apply(lambda x: x.day)
    df["intrinsic_value"] = df.apply(lambda x: OptionStats.calculate_intrinsic_value(x), axis=1)
    df["time_value"] = df.apply(lambda x: OptionStats.calculate_time_value(x), axis=1)
    # df = df.drop(["expirationDate"], axis=1)
    return df


def convertType(x):
    if x == 'C':
        return 'call'
    else:
        return 'put'

def find_strikes_min_max(all_strikes, strikeCount, stockPrice):
    index = 0
    all_strikes = np.sort(np.unique(all_strikes))
    for p in all_strikes:
        if p < stockPrice:
            index += 1
        else:
            break
    strike_min_index = index - np.int(strikeCount/2)
    if strike_min_index < 0:
        strike_min_index = 0
    strike_max_index = index + np.int(strikeCount/2)
    if strike_max_index >= len(all_strikes):
        strike_max_index = len(all_strikes)-1
    debug_print("min_max", all_strikes[strike_min_index], all_strikes[strike_max_index])
    return (all_strikes[strike_min_index], all_strikes[strike_max_index])


def filter_df_by_count(df, strikeCount, underlyingPrice):
    expDates = np.unique(df.exp_date)
    df_return = None
    for d in expDates:
        df_curr = df[df.exp_date == d]
        df_strikes = df_curr.Strike
        (min_strike, max_strike) = find_strikes_min_max(df_strikes, strikeCount,underlyingPrice)
        df_curr = df_curr[df_curr.Strike >= min_strike]
        df_curr = df_curr[df_curr.Strike <= max_strike]
        debug_print("d, shape", d, df_curr.shape)
        df_return = append_df(df_return, df_curr)
    return df_return


def check_option_symbol(x, symbol):
    if not x.OptionSymbol.startswith(symbol):
        print("wrong option symbol", x)
        exit(-1)

interest_rate = 0.03

def compute_iv_single_row(x):
    s = x.UnderlyingSymbol
    y = x.divYield
    sigma = read_one_stock_quote(s, x.data_date, "implied vol")

    if x.Type == "c":
        if y == 0:
            return newton_vol_call(x.UnderlyingPrice, x.Strike, x.days_to_expire, x.Bid, interest_rate, sigma)
        else:
            return newton_vol_call_div(x.UnderlyingPrice, x.Strike, x.days_to_expire, x.Bid, interest_rate, y, sigma)

    elif x.Type == "p":
        if y == 0:
            return newton_vol_put(x.UnderlyingPrice, x.Strike, x.days_to_expire, x.Bid, interest_rate, sigma)
        else:
            return newton_vol_put_div(x.UnderlyingPrice, x.Strike, x.days_to_expire, x.Bid, interest_rate, y, sigma)


def computeOptionHist(df, symbol):
    df.Type = df.Type.apply(lambda x: x.lower()[0:1])

    if "IV" not in df.columns:
        df["IV"] = df.apply(lambda x: compute_iv_single_row(x), axis=1)
        d_str = df.data_date[0].strftime("%Y%m%d")
        #df.to_csv('data' + os.sep + symbol + os.sep + d_str + "_option_chain.csv")

    df_call = df[df.Type == 'c']
    df_put = df[df.Type == 'p']

    df_call = df_call.sort_values(by=["days_to_expire", "Strike"])
    df_put = df_call.sort_values(by=["days_to_expire", "Strike"])

    df_call_aggr = OptionStats.groupby_columns_from_detail(df_call, OptionStats.aggregate_optionstats_from_detail)
    df_put_aggr = OptionStats.groupby_columns_from_detail(df_put, OptionStats.aggregate_optionstats_from_detail)
    df_call_aggr= df_call_aggr.reset_index()
    df_put_aggr= df_put_aggr.reset_index()
    #debug_print("df_call_aggr.columns", df_call_aggr.columns)
    df_total = pd.merge(df_call_aggr,df_put_aggr,\
                        left_on = ['UnderlyingSymbol', 'data_date'],\
                        right_on = ['UnderlyingSymbol', 'data_date'],\
                        suffixes=["_call", "_put"])
    df_total = df_total.rename(columns={"UnderlyingSymbol": "symbol", "data_date": "date"})
    debug_print("after computing optionstat history", df_total.shape, df_total.columns)
    return df_total

yield_map = {}
def get_stock_yield(symbol):
    if symbol in yield_map.keys():
        return yield_map[symbol]
    else:
        try:
            y = getStockDesc(symbol, "divYield")[0] / 100.  # divide the dividend yield by 100
            yield_map[symbol] = y
        except:
            y = 0
    return y
def persist_optionistics_file(df, symbol, d_cur, m):
    num_records_inserted = 0
    #debug_print(df.columns)
    symbols = np.unique(df[" under"])
    if len(symbols) != 1 or  symbols[0]!= symbol:
        return (0, 0)
    try:
        df = df.rename(columns={" put/call": "Type",
                                " volume": "Volume",
                                " open interest": "OpenInterest",
                                " delta": "Delta",
                                " theta": "Thelta",
                                " strike": "Strike",
                                " bid": "Bid",
                                " ask": "Ask",
                                " symbol": "OptionSymbol",
                                " expiration": "expirationDate",
                                " implied vol": "IV",
                                "date": "data_date"
                                })
    except KeyError:
        print("Error happens when renaming columns")
        exit(1)
    #debug_print("df.columns, df.shape", df.columns, df.shape)
    if recordExists('optionstat', {'$and': [{'symbol': {'$eq': symbol}}, {'date': {'$eq': d_cur}}]}):
        print("already exist", symbol, d_cur.strftime('%Y%m%d'))
        return 0
    stock_closing_price = read_one_stock_quote(symbol, d_cur, "close")
    if stock_closing_price is None:
        return 0
    df["UnderlyingPrice"] = stock_closing_price
    df["UnderlyingSymbol"] = symbol.upper()
    df["OptionSymbol"] = df["OptionSymbol"].apply(lambda x: x.replace(' ', ''))
    df.apply(lambda x: check_option_symbol(x, symbol.upper()), axis=1)
    df["Type"] = df.Type.apply(lambda x: convertType(x))
    df["exp_date"] = df["expirationDate"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df = derive_columns(df, d_cur)
    df = df[df.days_to_expire<= 90]
    df = df[df.exp_week == 3]
    df = filter_df_by_count(df, 30, stock_closing_price)
    y = get_stock_yield(symbol)
    df["divYield"] = y
    #debug_print(df.shape)
    df_curstat = computeOptionHist(df, symbol.upper())
    m.write_df(df_curstat, 'optionstat')
    num_stats_rec_inserted = df_curstat.shape[0]
    return num_stats_rec_inserted


def persist_td_option_file(df, symbol, d_cur, m):
    num_records_inserted = 0
    try:
        df = df.rename(columns={"putCall": "Type",
                                "totalVolume": "Volume",
                                "openInterest": "OpenInterest",
                                "last": "Last",
                                "bid": "Bid",
                                "delta": "Delta",
                                "theta": "Theta",
                                "strikePrice": "Strike",
                                "ask": "Ask",
                                "volatility": "IV"})
    except KeyError:
        print("error happens at renaming")
    if recordExists('optionstat', {'$and': [{'symbol':{'$eq': symbol}}, {'date':{'$eq': d_cur}}]}):
        print("already exist", symbol, d_cur.strftime('%Y%m%d'))
        return 0

    stock_closing_price = get_stock_price_history(symbol, d_cur)
    if stock_closing_price is None:
        return 0
    #debug_print("df.columns", df.columns)
    df["UnderlyingPrice"] = stock_closing_price
    df["UnderlyingSymbol"] = symbol
    df["Type"] = df.Type.apply(lambda x: x.lower())
    df["exp_date"] = df["expirationDate"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    df = derive_columns(df, d_cur)
    df["OptionSymbol"] = df.apply(lambda x: construct_option_symbol(x), axis=1)
    #debug_print(df.columns)
    #d_cur = datetime.datetime(d_cur.year, d_cur.month, d_cur.day)
    y = get_stock_yield(symbol)
    df["divYield"] = y
    df_stats_out = computeOptionHist(df, symbol)
    m.write_df(df_stats_out, 'optionstat')
    num_stats_rec_inserted = df_stats_out.shape[0]
    return num_stats_rec_inserted


def persist_option_dirs(dirName, symbols):
    debug_print("reading", dirName)
    m = mongo_api()
    num_stats_rec_inserted = 0
    for (root,dirs,files) in os.walk(dirName, topdown=True):
        if root == 'historical':
            continue
        files = np.sort(files)
        print("len files", len(files))
        for fi in files:
            (file_type, d_cur, curr_symbol, fi) = determine_file_origin(root, fi)
            print("file origin", file_type)
            if curr_symbol not in symbols:
                debug_print("skipped symbol ", curr_symbol)
                continue
            if file_type == 'TDA':
                print("processing:", fi)
                df = pd.read_csv(fi)
                x = persist_td_option_file(df, curr_symbol, d_cur, m)
            elif file_type == 'OPTIONISTICS':
                print("processing:", fi)
                df = pd.read_csv(fi)
                x = persist_optionistics_file(df, curr_symbol, d_cur,m)
            else:
                debug_print("skipping", fi)
            num_stats_rec_inserted += x
    print("number of records inserted to optionhist", num_stats_rec_inserted)


def find_dates(df_symbol, symbol):
    m = mongo_api()
    df_in_optionstats = m.read_df('optionstat', False, ["date"], [], {'symbol': {'$eq': symbol}},{'date': 1})
    date_set3 = set([])
    date_set2 = set(df_symbol.data_date)
    if df_in_optionstats.shape[0] >0:
        date_set3 = set(df_in_optionstats["date"])
    diff_2 = []
    if date_set3 != date_set2:
        diff_2 = date_set2 - date_set3
        diff_2 = np.sort(list(diff_2))
    del df_in_optionstats
    return diff_2


count_threshold = 35*3*2
def insert_one_symbol(df, s, m):
    debug_print("persist for symbol", s)
    df_cur_symbol = df[df.UnderlyingSymbol == s]
    dates_stats = find_dates(df_cur_symbol, s)
    df_stats_out = None
    num_d = 1
    for d in dates_stats:
        d_datetime = pd.to_datetime(d)
        cur_df = df_cur_symbol[df_cur_symbol.data_date == d_datetime]
        stock_closing_price = np.min(cur_df["UnderlyingPrice"])
        cur_df = cur_df[cur_df.exp_week == 3]
        cur_df = cur_df[cur_df.days_to_expire <= 90]
        cur_df = filter_df_by_count(cur_df, 30, stock_closing_price)
        df_cur_stat = computeOptionHist(cur_df, s)
        df_stats_out = append_df(df_stats_out, df_cur_stat)
        num_d += 1
    if df_stats_out is not None and df_stats_out.shape[0] > 0:
        m.write_df(df_stats_out, 'optionstat')
        x = df_stats_out.shape[0]
        del df_stats_out
        return x
    return 0

def persist_option_hist_file(fileName, symbols):
    df = pd.read_pickle(fileName)
    #debug_print(df.columns)
    num_stats = 0
    if df is not None:
        m = mongo_api()
        df = df.drop("Expiration", axis=1)
<<<<<<< HEAD
        debug_print("df.shape", df.shape)
        for s in symbols:
            try:
                numstats_per_symbol = insert_one_symbol(df, s, m)
                debug_print(s, ":", numstats_per_symbol)
                num_stats += numstats_per_symbol
            except:
                print("Error when processing symbol", s)
                continue
    debug_print("number of records inserted to optionhist", num_stats)

def load_json_for_symbol(symbol):
    symbols = []
    symbol = symbol.replace("'", "\"")
    symbol = '{\"symbols\":' + symbol + '}'
    if symbol is not None:
        symbols = json.loads(symbol)
        return symbols["symbols"]
    return None
=======
        if symbol is not None:
            df = df[df.UnderlyingSymbol == symbol]
        dates = np.unique(df.data_date)
        print("df.shape", df.shape)
        for d in dates:
            cur_df = df[df.data_date == d]
            stock_closing_price = np.min(cur_df["UnderlyingPrice"])
            cur_df = filter_df_by_count(cur_df, 50, stock_closing_price)
            m.write_df(cur_df, "optionhist")

>>>>>>> bbba58a92acd0e075c0b271c3f54bb4d4623e4ed

def main(argv):
    fileName = None
    dirName = None
    symbol = None
    symbols = []
    try:
        opts, args = getopt.getopt(argv, "hf:d:s:w:",
                                   ["help", "f=", "d=", "s="])
        for opt, arg in opts:
            if opt == '-h':
                print(sys.argv[0] + '-f <filename> -d <dirName> -s <symbol> -w <watch_list>')
                sys.exit()
            if opt == '-f':
                print("getting file", arg)
                fileName = arg
            if opt == '-d':
                dirName = arg
            if opt == '-s':
                symbol = arg
                symbols = load_json_for_symbol(symbol)

            if opt == '-w':
                watch_list_file = arg
                symbols = load_watch_lists(watch_list_file)
        if symbols is not None:
            symbols = np.sort(symbols)
        if len(symbols) == 0:
            print("please specifiy symbol as -s or -w")
            exit(-1)
        if fileName is not None and os.path.exists(fileName):
            persist_option_hist_file(fileName, symbols)
        elif fileName is not None and not os.path.exists(fileName):
            print("Can't find", fileName)
            exit(-1)
<<<<<<< HEAD
=======
        if fileName is not None and os.path.exists(fileName):
            persist_option_hist_file(fileName, symbol)
        elif fileName is not None and not os.path.exists(fileName):
            print("Can't find", fileName)
            exit(-1)
>>>>>>> bbba58a92acd0e075c0b271c3f54bb4d4623e4ed
        if dirName is not None:
            try:
                persist_option_dirs(dirName, symbols)
            except KeyError:
                print("Error when reading option_dirs")
                exit(-1)
    except TypeError:
        print("Error in argument")
        exit(-1)

if __name__ == "__main__":
    main(sys.argv[1:])
