# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\Umeda\python\stockML\tda_api.py
# Size of source mod 2**32: 8582 bytes
import tda_authenticate, json
from requests import Session, Request, PreparedRequest
import pandas as pd, numpy as np, requests
from pandas.io.json import json_normalize
from datetime import datetime
from os import path
import time
from monthdelta import monthdelta

class Td:
    access_token = ''
    refresh_token = ''
    apiKey = ''
    client_id = ''
    symbol = ''
    access_token_timestamp = ''
    access_token_expires_in = 0
    refresh_token_expires_in = 0

    def __init__(self, access_token, refresh_token, client_id, apikey, access_token_timestamp, access_token_expires_in, refresh_token_expires_in):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.apikey = apikey
        self.client_id = client_id
        self.access_token_timestamp = access_token_timestamp
        self.access_token_expires_in = access_token_expires_in
        self.refresh_token_expires_in = refresh_token_expires_in

    @staticmethod
    def get_access_file_name():
        token_file = 'access_token.json'
        return token_file

    @staticmethod
    def convert_timestamp_to_time(ts, param):
        return pd.to_datetime(ts, unit=param)

    @staticmethod
    def dump_json_File(res, f):
        with open(f, 'w') as (tFile):
            json.dump(res, tFile)

    @staticmethod
    def load_json_file(f):
        with open(f, 'r') as (tFile):
            data = json.load(tFile)
            return data

    @staticmethod
    def update_key_value_in_file(f, params):
        data = Td.load_json_file(f)
        for k in params.keys():
            data[k] = params[k]

        Td.dump_json_File(data, f)

    @staticmethod
    def diff_time(dt1, dt2):
        return int(dt1 - dt2)

    def check_access_token_valid(self):
        currTimestamp = time.time()
        if self.diff_time(currTimestamp, self.access_token_timestamp) > self.access_token_expires_in:
            res = tda_authenticate.refresh_token(self.refresh_token, self.client_id)
            print('refreshing tokens to', res)
            self.access_token = res['access_token']
            self.access_token_timestamp = currTimestamp
            params = dict({'access_token':self.access_token,  'access_token_timestamp':currTimestamp})
            Td.update_key_value_in_file(Td.get_access_file_name(), params)

    def get_quotes(self, symbol):
        self.check_access_token_valid()
        access_token = self.access_token
        headers = {'Content-Type':'application/x-www-form-urlencoded',  'Authorization':'Bearer {}'.format(access_token)}
        data = {'symbol':symbol,  'apikey':self.apikey}
        authReply = requests.get('https://api.tdameritrade.com/v1/marketdata/quotes', headers=headers,
          params=data)
        return authReply.json()

    def get_price_history(self, symbol, startDate=None, endDate=None, option_params={'periodType':'year', 
 'frequency':'monthly', 
 'period':1, 
 'frequency':1}):
        self.check_access_token_valid()
        access_token = self.access_token
        headers = {'Content-Type':'application/x-www-form-urlencoded',  'Authorization':'Bearer {}'.format(self.access_token)}
        periodType = option_params['periodType']
        frequencyType = option_params['frequencyType']
        period = option_params['period']
        frequency = option_params['frequency']
        data = {'periodType':periodType,  'frequencyType':frequencyType,  'period':period, 
         'frequency':frequency,  'startDate':startDate, 
         'endDate':endDate}
        authReply = requests.get(('https://api.tdameritrade.com/v1/marketdata/' + symbol + '/pricehistory'), headers=headers,
          params=data)
        candles = authReply.json()
        df = json_normalize(authReply.json())
        df = pd.DataFrame(candles['candles'])
        df['datetime'] = Td.convert_timestamp_to_time(df['datetime'], 'ms')
        return df

    def get_option_chain(self, option_params):
        print('getting options, parameters:', option_params)
        monthlyOnly = option_params['monthlyOption']
        today_date = datetime.today()
        self.check_access_token_valid()
        token = self.access_token
        headers = {'Content-Type':'application/x-www-form-urlencoded',  'Authorization':'Bearer {}'.format(token)}
        data = {'symbol':option_params['symbol'],  'api_key':self.apiKey,  'contractType':option_params['contractType'], 
         'strikeCount':option_params['strikeCount'], 
         'includeQuotes':True, 
         'strike':option_params['strike'],  'range':'S', 
         'fromDate':today_date,  'toDate':today_date + monthdelta(option_params['monthIncrement']), 
         'optionType':'S'}
        authReply = requests.get('https://api.tdameritrade.com/v1/marketdata/chains', headers=headers,
          params=data)
        optionJson = authReply.json()
        Td.dump_json_File(optionJson, 'option_json')
        oc = ''
        if option_params['contractType'] == 'CALL':
            oc = optionJson['callExpDateMap']
        if option_params['contractType'] == 'PUT':
            oc = optionJson['putExpDateMap']
        exprDays = oc.keys()
        df = pd.DataFrame()
        for e in exprDays:
            v = oc[e]
            strikes = v.keys()
            for s in strikes:
                r = list()
                values = np.array(v[s]).flatten()
                r.append(e)
                r.append(s)
                isWeekly = False
                for a in values:
                    if monthlyOnly == True:
                        if 'Weekly' in a['description']:
                            isWeekly = True
                        for optionAttr in option_params['interested_columns']:
                            if optionAttr in a.keys():
                                r.append(a[optionAttr])

                if isWeekly:
                    if monthlyOnly:
                        continue
                df = df.append((pd.Series(r)), ignore_index=True)

        df.columns = option_params['interested_columns']
        return df

    @staticmethod
    def init(user, secrete):
        config_file = 'config.json'
        access_token_file = 'access_token.json'
        data = []
        access_token = ''
        client_id = ''
        redirect_url = ''
        tda_user = ''
        tda_password = ''
        data = Td.load_json_file(config_file)
        client_id = data['API_KEY']
        redirect_url = 'https://' + data['HOST']
        access_token = ''
        access_token_timestamp = ''
        access_token_expires_in = 0
        refresh_token_expires_in = 0
        refresh_token = ''
        if path.exists(access_token_file):
            tokens = Td.load_json_file(access_token_file)
            access_token = tokens['access_token']
            refresh_token = tokens['refresh_token']
            access_token_timestamp = tokens['access_token_timestamp']
            access_token_expires_in = tokens['expires_in']
            refresh_token_expires_in = tokens['refresh_token_expires_in']
        else:
            try:
                res = tda_authenticate.authentication(client_id, redirect_url, user, secrete)
                access_token = res['access_token']
                refresh_token = res['refresh_token']
                print(time.time())
                res['access_token_timestamp'] = time.time()
                Td.dump_json_file(res, Td.get_access_file_name())
                access_token_expires_in = res['expires_in']
                refresh_token_expires_in = res['refresh_token_expires_in']
            except BaseException:
                print("can't open config.json file")

        td = Td(access_token, refresh_token, client_id, data['API_KEY'], access_token_timestamp, access_token_expires_in, refresh_token_expires_in)
        return td
# okay decompiling tda_api.pyc
