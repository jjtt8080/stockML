import pandas as pd
import numpy as np
import os as os
import sys as sys
import datetime
from util import debug_print
sys.path.append("..")
from trade_api.mongo_api import mongo_api
from trade_api.tda_api import Td

def getStockDesc(symbols, descName):
    m = mongo_api()
    if type(symbols) == str:
        symbols = [symbols]
    filter = {"symbol": {'$in': symbols}}
    df = m.read_df('stock_quotes', False, ['symbol', descName], [], filter, {"symbol":1})
    assert(df.shape[0] == len(symbols))
    i = list(df.columns).index(descName)
    return df.iloc[:,i]

def recordExists(collection_name, filter):
    m = mongo_api()
    bexist = (m.count(collection_name, filter) > 0)
    return bexist

def construct_day_filter(currDate):
    if currDate.isoweekday() == 6:
        currDate = currDate - datetime.timedelta(1)
    if currDate.isoweekday() == 7:
        currDate = currDate - datetime.timedelta(2)

    date_filter = {'$and': [{'year': {'$eq': currDate.year}},\
                            {'month': {'$eq': currDate.month}},\
                            {'d_index': {'$eq': currDate.day}}]}
    return date_filter


def read_one_stock_quote(symbol,currDate, projection):
    m = mongo_api()
    date_filter = construct_day_filter(currDate)
    assert(type(projection) == str)
    try:
        db_df = m.read_df('stockcandles', False, [projection], [], {'$and': [{'symbol': {'$eq': symbol}}, date_filter]}, None)
        if db_df.shape is not None and db_df.shape[0] >= 1:
            debug_print("found price for ", currDate, " at ", db_df.iloc[0,1])
            return db_df.iloc[0, 1]
        else:
            print("can't find specific symbol", symbol)
            return None
    except KeyError:
        print("error when reading single stock price history")
        return None


def get_stock_price_history(symbol, currDate):
    try:
        p = read_one_stock_quote(symbol,currDate, "close")
        if p is None:
            debug_print("can't find the symbol in specific date", symbol, currDate)
        return p
    except TypeError:
        print("error when get stock price history")
        exit(1)


def check_persist_timing(m, collection_name, date_filter, today):
    if Td.is_trading_day(today) and not Td.is_market_open(today) and m.countDistinct(collection_name, date_filter) == 0:
        print("Persisting due to 0 records in db for today")
        return True
    else:
        print("Not persisting due to either market is not open, or currently is market hour, or there is already record in db")
        return False