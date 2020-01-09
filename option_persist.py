# This file persist the option detail data we obtained from optiondata website
# to the optionhist collection in mongo
#  it also persist the data we daily grab from tda
import pandas as pd
import numpy as np
import os as os
import sys as sys
from trade_api.mongo_api import mongo_api
import getopt
import csv
import datetime
from analyze_historical import OptionStats
from optionML import getResultByType
from trade_api.tda_api import Td
import timedelta
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
    print("determine file origin", fi)
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


def converType(x):
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
    print("min_max", all_strikes[strike_min_index], all_strikes[strike_max_index])
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
        print("d, shape", d, df_curr.shape)
        if df_return is None:
            df_return = df_curr
        else:
            df_return = df_return.append(df_curr, ignore_index=True)
    return df_return


def check_option_symbol(x, symbol):
    if not x.OptionSymbol.startswith(symbol):
        print("wrong option symbol", x)
        exit(-1)


def persist_optionistics_file(df, symbol, d_cur, m):
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
                                "date": "data_date"
                                }, inplace=True)
    except KeyError:
        print("Error happens when renaming columns")
        exit(1)
    debug_print("df.columns, df.shape", df.columns, df.shape)
    stock_closing_price = get_stock_price_history(symbol, d_cur)
    if stock_closing_price is None:
        return
    df["UnderlyingPrice"] = stock_closing_price
    df["UnderlyingSymbol"] = symbol.upper()
    df["OptionSymbol"] = df["OptionSymbol"].apply(lambda x: x.replace(' ', ''))
    df.apply(lambda x: check_option_symbol(x, symbol.upper()), axis=1)
    df["Type"] = df.Type.apply(lambda x: converType(x))
    df["exp_date"] = df["expirationDate"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df = derive_columns(df, d_cur)
    df = df[df.days_to_expire<= 90]
    df = df[df.exp_week == 3]
    df = filter_df_by_count(df, 50, stock_closing_price)
    debug_print(df.shape)
    if m.count('optionhist', {'$and': [{'UnderlyingSymbol': {'$eq': symbol}}, {'data_date': {'$eq': d_cur}}]}) >0:
        print("avoid duplicate, not inserting", d_cur)
        return
    m.write_df(df, 'optionhist')


def persist_td_option_file(df, symbol, d_cur, m):
    try:
        df = df.rename(columns={"putCall": "Type",
                                "totalVolume": "Volume",
                                "openInterest": "OpenInterest",
                                "last": "Last",
                                "bid": "Bid",
                                "delta": "Delta",
                                "theta": "Theta",
                                "strikePrice": "Strike",
                                "ask": "Ask"})
    except KeyError:
        print("error happens at renaming")
    stock_closing_price = get_stock_price_history(symbol, d_cur)
    if stock_closing_price is None:
        return
    debug_print("df.columns", df.columns)
    df["UnderlyingPrice"] = stock_closing_price
    df["UnderlyingSymbol"] = symbol
    df["Type"] = df.Type.apply(lambda x: x.lower())
    df["exp_date"] = df["expirationDate"].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    df = derive_columns(df, d_cur)
    df["OptionSymbol"] = df.apply(lambda x: construct_option_symbol(x), axis=1)
    debug_print(df.columns)
    d_cur = datetime.datetime(d_cur.year, d_cur.month, d_cur.day)
    count= m.count('optionhist', {'$and': [{'UnderlyingSymbol': {'$eq': symbol}}, {'data_date': {'$eq': d_cur}}]})
    if count > 0:
        print("avoid duplicate, not inserting", d_cur)
        return
    print(df)
    m.write_df(df, 'optionhist')


def persist_option_dirs(dirName, symbol):
    print("reading", dirName)
    m = mongo_api()
    lines_inserted = persist_stock_price_history(symbol)
    debug_print("line_inserted", lines_inserted)
    for (root,dirs,files) in os.walk(dirName, topdown=True):
        for fi in files:
            (file_type, d_cur, curr_symbol, fi) = determine_file_origin(root, fi)
            if curr_symbol != symbol:
                continue
            if file_type == 'TDA':
                df = pd.read_csv(fi)
                persist_td_option_file(df, curr_symbol, d_cur, m)
            elif file_type == 'OPTIONISTICS':
                df = pd.read_csv(fi)
                persist_optionistics_file(df, curr_symbol, d_cur,m)
            else:
                print("skipping", fi)


def persist_option_hist_file(fileName, symbol):
    df = pd.read_pickle(fileName)
    debug_print(df.columns)
    if df is not None:
        m = mongo_api()
        #df = df.drop("IV", axis=1)
        df = df.drop("Expiration", axis=1)
        if symbol is not None:
            df = df[df.UnderlyingSymbol == symbol]
        dates = np.unique(df.data_date)
        print("df.shape", df.shape)
        for d in dates:
            cur_df = df[df.data_date == d]
            stock_closing_price = np.min(cur_df["UnderlyingPrice"])
            cur_df = filter_df_by_count(cur_df, 50, stock_closing_price)
            m.write_df(cur_df, "optionhist")


def main(argv):
    fileName = None
    dirName = None
    symbol = None
    try:
        opts, args = getopt.getopt(argv, "hf:d:s:",
                                   ["help", "f=", "d=", "s="])
        for opt, arg in opts:
            if opt == '-h':
                print(sys.argv[0] + '-f <filename> -d <dirName> -s <symbol>')
                sys.exit()
            if opt == '-f':
                print("getting file", arg)
                fileName = arg
            if opt == '-d':
                dirName = arg
            if opt == '-s':
                symbol = arg
        if symbol is None:
            print("please specifiy symbol as -s")
            exit(-1)
        if fileName is not None and os.path.exists(fileName):
            persist_option_hist_file(fileName, symbol)
        elif fileName is not None and not os.path.exists(fileName):
            print("Can't find", fileName)
            exit(-1)
        if dirName is not None:
            try:
                persist_option_dirs(dirName, symbol)
            except KeyError:
                print("Error when reading option_dirs")
                exit(-1)
    except TypeError:
        print("Error in argument")
        exit(-1)

if __name__ == "__main__":
    main(sys.argv[1:])
