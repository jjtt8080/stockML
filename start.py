import pandas as pd
import numpy as np
import numpy.ma as ma
from trade_api.tda_api import Td
from trade_api.mongo_api import mongo_api
import os as os
import sys as sys
import json
import csv
import datetime
from analyze_historical import OptionStats
from optionML import getResultByType
import timedelta
from util import debug_print, append_df, load_watch_lists
from trade_api.db_api import read_one_stock_quote, getStockDesc, get_stock_price_history, recordExists
from trade_api.calculate_iv import newton_vol_call,newton_vol_put,newton_vol_call_div,newton_vol_put_div

m = mongo_api()

def readTable(tableName, projection=None):
    if projection is None:
        projection = '*'
    df = m.read_df(tableName, False, projection, [], {}, {})
    return df

def getMaxMin(df, colName):
    return np.max(df[colName]), np.min(df[colName])

def aggr(df, projCol, expression):
    gb = df.groupby(by=projCol, as_index=True)
    return gb.aggregate(expression)

def date_filter(df, colName, min_year, max_year, min_month=1, max_month=12, min_day=1, max_day=31):
    return_df = df[(df[colName] >= datetime.datetime(min_year, min_month, min_day))]
    print(return_df.shape)
    return_df = return_df[return_df[colName] < datetime.datetime(max_year, max_month, max_day)]
    return return_df


def yesterday():
    #today_ts = np.int((pd.Timestamp(dates[0]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    today = datetime.datetime.today()
    prev_trading = Td.get_prev_trading_day(today)
    return prev_trading


