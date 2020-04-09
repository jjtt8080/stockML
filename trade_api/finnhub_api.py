# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\Umeda\python\stockML\finnhub_api.py
# Size of source mod 2**32: 1485 bytes
import requests
import json
from datetime import date
from datetime import time
import datetime
import pandas as pd
import numpy as np
import csv
import os
from pandas.io.json import json_normalize

DEBUG = 4
INFO = 2
NONE = 0
g_interestedColumns =  ['expirationDate', 'strike','volume', 'openInterest', 'lastPrice', 'bid', 'ask', 'delta', 'lastTradeDateTime']

class Finnapi:
    api_key = ''
    curr_symbol = ''
    request_string = ''
    response = ''
    log_level = NONE


    def __init__(self, key, loglevel):
        if key == '':
            path = os.path.dirname(os.path.realpath(__file__)) + os.sep
            with open(path + 'config/finconfig.json', 'r') as (tFile):
                data = json.load(tFile)
                key = data["API_KEY"]
        self.api_key = key
        self.log_level = loglevel


    @staticmethod
    def any_non_type(item, list_of_attr):
        for l in list_of_attr:
            if (item[l] is None):
                return True
        return False

    @staticmethod
    def convertAttrToFloat(item, list_of_attr):
        r = item
        for l in list_of_attr:
            r[l] = np.float(item[l])
        return r

    @staticmethod
    def filter_options_df(res, expiration_date, option_type, openInterest_min, volume_min, interestedColumns=g_interestedColumns):
        df = pd.DataFrame()
        for item in res['data']:
            if (expiration_date == None or item['expirationDate'] == expiration_date):
                ## option_type can be 'CALL' or 'PUT'
                ops = item['options'][option_type]
                for o in ops:
                    r = list();
                    if (Finnapi.any_non_type(o, interestedColumns)):
                        continue
                    o = Finnapi.convertAttrToFloat(o, interestedColumns)
                    for optionAttr in interestedColumns:
                        r.append(a[optionAttr])
                    df.append(pd.Series(r), ignoreIndex=True)
        df.columns = interestedColumns;
        return df;
                        
    @staticmethod
    def filter_options(res, expiration_date, option_type, strike_low, strike_high, openInterest_min, volume_min):
        s_list = []
        oi_list = []
        v_list = []
        p_list = []
        b_list = []
        a_list = []
        for item in res['data']:
            if (expiration_date == None or item['expirationDate'] == expiration_date):
                ## option_type can be 'CALL' or 'PUT'
                ops = item['options'][option_type]
                for o in ops:
                    if (Finnapi.any_non_type(o, ['strike', 'volume', 'openInterest', 'bid', 'ask', 'lastPrice'])):
                        continue
                    o = Finnapi.convertAttrToFloat(o, ['strike', 'volume', 'openInterest', 'bid', 'ask', 'lastPrice'])
                    s = np.float(o['strike'])
                    if ((s >= strike_low) and s <= strike_high and
                            o['openInterest'] > openInterest_min and o['volume'] > volume_min):
                        s_list.append(s)
                        oi_list.append(o['openInterest'])
                        v_list.append(o['volume'])
                        p_list.append(o['lastPrice'])
                        b_list.append(o['bid'])
                        a_list.append(o['ask'])
        return (s_list, oi_list, v_list, p_list, b_list, a_list)


    def log(self, cur_loglevel, msg):
        if cur_loglevel <= self.log_level:
            print(msg)

    def call_api(self, api_action, params, result_type='json'):
        self.log(DEBUG, 'call_api')
        self.request_string = 'https://finnhub.io/api/v1/' + api_action + '?symbol=' + self.curr_symbol
        self.request_string += '&'
        if params != '':
            for k in params.keys():
                self.request_string += k
                self.request_string += '='
                self.request_string += str(params[k])
                self.request_string += '&'

        self.request_string += 'token='
        self.request_string += self.api_key
        self.log(DEBUG, self.request_string)
        print("request_string" + self.request_string)
        self.response = requests.get(self.request_string)
        self.log(DEBUG, self.response)
        res = self.response.json()
        return res

    def getOptionChain(self):
        self.log(INFO, 'Getting option chain for symbol ' + self.curr_symbol)
        params = {}
        self.log(DEBUG, params.keys())
        res = self.call_api('stock/option-chain', params)
        return res

    def getCandleStick(self, symbol, option_params):
        param = json.loads(option_params)
        self.curr_symbol = symbol
        print(param)
        response = self.call_api('stock/candle',  param)
        if response["s"] != 'ok':
            return None
        df = pd.DataFrame(data = response)
        return df

    def getUpDownGrade(self, symbol):
        self.curr_symbol = symbol
        response = self.call_api('stock/upgrade-downgrade', '')
        df = pd.DataFrame(data = response)
        df = df.sort_values('gradeTime')
        return df



    def getPriceTarget(self, symbol):
        self.curr_symbol = symbol
        response = self.call_api('stock/price-target', '')
        df = pd.DataFrame.from_dict(response, orient='index').reset_index().transpose()
        return df

    def getRecommendations(self, symbol):
        self.curr_symbol = symbol
        response = self.call_api('stock/recommendation', '')
        df = pd.DataFrame(response)
        df = df.sort_values('period').reset_index()
        return df

    def getPatterns(self, symbol, params):
        self.curr_symbol = symbol
        response = self.call_api('scan/pattern', params)
        df = pd.DataFrame(data=response)
        return df

    def getTechInd(self, symbol, params):
        self.curr_symbol = symbol
        response = self.call_api('/indicator', params)
        df = pd.DataFrame(data=response)
        df["symbol"] = symbol
        today = date.today()
        df["date"] = today
        return df

    def getEarnings(self,symbol, params, earning_type):
        self.curr_symbol = symbol
        if earning_type == 'rev-est':
            response = self.call_api('stock/revenue-estimate', params)
            df = json_normalize(response["data"])
        elif earning_type == 'eps-est':
            response = self.call_api('stock/eps-estimate', params)
            df = json_normalize(response["data"])
        elif earning_type == 'earnings':
            response = self.call_api('stock/earnings', params)
            df = json_normalize(response)
            df = df.sort_values('period').reset_index()

        df["symbol"] = symbol
        return df



# okay decompiling finnhub_api.pyc
