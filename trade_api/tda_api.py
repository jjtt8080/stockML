# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\Umeda\python\stockML\tda_api.py
# Size of source mod 2**32: 8582 bytes
import json
import pandas as pd, numpy as np, requests
from pandas.io.json import json_normalize
from datetime import datetime
from os import path
import time
from monthdelta import monthdelta
import os as os, os.path, sys, requests, time
from selenium import webdriver
from shutil import which
from urllib import parse as up
import time
from datetime import datetime
from dateutil.tz import tzutc
import pandas_market_calendars as mcal


class Td:
    access_token = ''
    refresh_token = ''
    apiKey = ''
    client_id = ''
    symbol = ''
    access_token_timestamp = 0.0
    access_token_expires_in = 0
    refresh_token_expires_in = 0

    def __init__(self, access_token, refresh_token, client_id, apikey, access_token_timestamp, access_token_expires_in, refresh_token_expires_in):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.apiKey = apikey
        self.client_id = client_id
        self.access_token_timestamp = access_token_timestamp
        self.access_token_expires_in = access_token_expires_in
        self.refresh_token_expires_in = refresh_token_expires_in

    @staticmethod
    def authentication(client_id, redirect_uri, tdauser=None, tdapass=None):
        client_id = client_id + '@AMER.OAUTHAP'
        #print('client_id:', client_id)
        url = 'https://auth.tdameritrade.com/auth?response_type=code&redirect_uri=' + up.quote(
            redirect_uri) + '&client_id=' + up.quote(client_id)
        options = webdriver.ChromeOptions()
        if sys.platform == 'darwin':
            if os.path.exists('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'):
                options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
            elif os.path.exists('/Applications/Chrome.app/Contents/MacOS/Google Chrome'):
                options.binary_location = '/Applications/Chrome.app/Contents/MacOS/Google Chrome'
            elif 'linux' in sys.platform:
                options.binary_location = which('google-chrome') or which('chrome') or which('chromium')
        if os.path.exists('C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'):
            options.binary_location = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
            options.binary_location = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
        else:
            chrome_driver_binary = which('chromedriver') or '/usr/local/bin/chromedriver'
            driver = webdriver.Chrome(chrome_driver_binary, chrome_options=options)
            #print('found driver', driver)
            driver.get(url)

            tdauser = tdauser or os.environ.get('TDAUSER', '')
            tdapass = tdapass or os.environ.get('TDAPASS', '')

        if tdauser:
            if tdapass:
                ubox = driver.find_element_by_id('username')
                pbox = driver.find_element_by_id('password')

                ubox.send_keys(tdauser)
                pbox.send_keys(tdapass)
                driver.find_element_by_id('accept').click()
                time.sleep(2)
                driver.find_element_by_id('accept').click()
                while True:
                    try:
                        code = up.unquote(driver.current_url.split('code=')[1])
                        if code != '':
                            break
                        else:
                            time.sleep(2)
                    except (TypeError, IndexError):
                        pass

            else:
                input('after giving access, hit enter to continue')
                code = up.unquote(driver.current_url.split('code=')[1])
            driver.close()
            resp = requests.post('https://api.tdameritrade.com/v1/oauth2/token',
                                 headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                 data={'grant_type': 'authorization_code',
                                       'refresh_token': '',
                                       'access_type': 'offline',
                                       'code': code,
                                       'client_id': client_id,
                                       'redirect_uri': redirect_uri})
            assert not resp.status_code != 200, 'Could not authenticate!'
            #print(resp.json())
            return resp.json()

    @staticmethod
    def refresh_tokens(refresh_token, client_id):
        resp = requests.post('https://api.tdameritrade.com/v1/oauth2/token',
                             headers={'Content-Type': 'application/x-www-form-urlencoded'},
                             data={'grant_type': 'refresh_token',
                                   'refresh_token': refresh_token,
                                   'client_id': client_id})
        assert not resp.status_code != 200, 'Could not authenticate!'
        return resp.json()

    @staticmethod
    def get_script_dir():
        return os.path.dirname(os.path.realpath(__file__)) + os.sep

    @staticmethod
    def get_tmp_dir():
        return Td.get_script_dir() + "tmp" + os.sep

    @staticmethod
    def get_config_dir():
        return Td.get_script_dir() + "config" + os.sep

    @staticmethod
    def get_access_file_name():
        token_file = Td.get_tmp_dir() + "access_token.json"
        return token_file

    @staticmethod
    def get_epoch_ms(time_input):
        if time_input is None:
            ms = int(round(time.time() * 1000))
            return ms
        else:
            dt_2020 = datetime.fromisoformat(time_input)
            ms = dt_2020.strftime('%d')
            return ms

    @staticmethod
    def convert_timestamp_to_time(ts, param):
        return pd.to_datetime(ts, unit=param)

    @staticmethod
    def dump_json_file(res, f):
        with open(f, 'w') as (tFile):
            json.dump(res, tFile)

    @staticmethod
    def load_json_file(f):
        #print("opening", f)
        with open(f, 'r') as (tFile):
            data = json.load(tFile)
            return data

    @staticmethod
    def update_key_value_in_file(f, params):
        data = Td.load_json_file(f)
        for k in params.keys():
            data[k] = params[k]

        Td.dump_json_file(data, f)



    @staticmethod
    def diff_time(dt1, dt2):
        return int(dt1 - dt2)

    def check_access_token_valid(self):
        currTimestamp = time.time()
        if self.diff_time(currTimestamp, self.access_token_timestamp) > self.access_token_expires_in:
            res = Td.refresh_tokens(self.refresh_token, self.client_id)
            #print('refreshing tokens to', res['access_token'])
            self.access_token = res['access_token']
            self.access_token_timestamp = currTimestamp
            params = dict({'access_token':self.access_token,  'access_token_timestamp':currTimestamp})
            Td.update_key_value_in_file(Td.get_access_file_name(), params)

    def get_quotes(self, symbol):
        #print("api_key", self.client_id)
        self.check_access_token_valid()
        access_token = self.access_token
        headers = {'Content-Type':'application/x-www-form-urlencoded',  'Authorization':'Bearer {}'.format(access_token)}
        data = {'symbol':symbol,  'apikey':self.apiKey}

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
        #periodType = option_params['periodType']
        #frequencyType = option_params['frequencyType']
        #period = option_params['period']
        #frequency = option_params['frequency']
        #if startDate is None:
        #    startDate = Td.get_epoch_ms(startDate)

        data = option_params
        if startDate is not None:
            data['startDate'] = startDate,
        if endDate is not None:
            data['endDate'] = endDate
        authReply = requests.get(('https://api.tdameritrade.com/v1/marketdata/' + symbol + '/pricehistory'), headers=headers,
          params=data)
        candles = authReply.json()
        #print("price_history result json", candles)
        df = json_normalize(authReply.json())
        #print("df",df)
        df = pd.DataFrame(candles['candles'])
        df['datetime'] = Td.convert_timestamp_to_time(df['datetime'], 'ms')
        return df


    def get_option_chain_map(self, oc, exprDays, monthlyOnly, option_params):
        df = pd.DataFrame()
        for e in exprDays:
            v = oc[e]
            strikes = v.keys()
            for s in strikes:
                r = list()
                values = np.array(v[s]).flatten()
                isWeekly = False
                for a in values:
                    #print(a.keys())
                    if monthlyOnly == True:
                        if 'Weekly' in a['description']:
                            isWeekly = True
                    if isWeekly and monthlyOnly:
                        continue
                    for optionAttr in option_params['interested_columns']:
                        if optionAttr in a.keys():
                            r.append(a[optionAttr])
                        else:
                            print("unmatched column name", optionAttr)

                if isWeekly and monthlyOnly:
                    continue
                df = df.append((pd.Series(r)), ignore_index=True)
        return df


    def get_option_chain(self, option_params):
        #print('getting options, parameters:', option_params)
        monthlyOnly = option_params['monthlyOption']
        today_date = datetime.today()
        self.check_access_token_valid()
        token = self.access_token
        headers = {'Content-Type':'application/x-www-form-urlencoded',  'Authorization':'Bearer {}'.format(token)}
        data = {'symbol':option_params['symbol'],  'apikey':self.apiKey,  'contractType':option_params['contractType'],
         'strikeCount':option_params['strikeCount'], 
         'includeQuotes':True, 
         'strike':option_params['strike'],  'range':'S', 
         'fromDate':today_date,  'toDate':today_date + monthdelta(option_params['monthIncrement']), 
         'optionType':'S'}
        #print(data)
        authReply = requests.get('https://api.tdameritrade.com/v1/marketdata/chains', headers=headers,
          params=data)
        optionJson = authReply.json()
        Td.dump_json_file(optionJson, Td.get_tmp_dir() + 'option_json')
        oc = ''
        if option_params['contractType'] == 'CALL':
            oc = optionJson['callExpDateMap']
            exprDays = oc.keys()
            df = self.get_option_chain_map(oc, exprDays, monthlyOnly, option_params)
        if option_params['contractType'] == 'PUT':
            oc = optionJson['putExpDateMap']
            exprDays = oc.keys()
            df = self.get_option_chain_map(oc, exprDays, monthlyOnly, option_params)
        if option_params['contractType'] == 'ALL':
            oc = optionJson['callExpDateMap']
            exprDays = oc.keys()
            df = self.get_option_chain_map(oc, exprDays, monthlyOnly, option_params)
            oc = optionJson['putExpDateMap']
            exprDays = oc.keys()
            df2 = self.get_option_chain_map(oc, exprDays, monthlyOnly, option_params)
            if df.shape[0] > 0:
                df = df.append(df2)
            else:
                df = df2;

        if df is not None and df.shape[1] == len(option_params['interested_columns']):
            df.columns = option_params['interested_columns']
        else:
            if df is not None:
                print("option chain shape, not the same as interested_columns", df.shape)
            return None;
        return df

    @staticmethod
    def is_market_open(t):
        inputD = datetime.fromtimestamp(t)
        thisDate = datetime.date(inputD)
        c = Td.get_market_hour(thisDate, thisDate)
        if c.shape[0] == 1:
            return inputD >= c["market_open"][0] and inputD < c["market_close"][0]
        return False

    @staticmethod
    def get_market_hour(s, e):
        nyse = mcal.get_calendar('NYSE')
        schedules = nyse.schedule(start_date=s, end_date=e)
        schedules["market_day"] = schedules["market_open"].apply(lambda x: datetime(x.year, x.month, x.day, 0, 0, 0))
        return schedules

    @staticmethod
    def init(user, secrete):
        config_file = Td.get_config_dir() + 'config.json'
        access_token_file = Td.get_tmp_dir() + 'access_token.json'
        data = []
        access_token = ''
        client_id = ''
        redirect_url = ''
        tda_user = ''
        tda_password = ''
        data = Td.load_json_file(config_file)
        client_id = data['API_KEY']
        if user == '':
            user = data['USER']
        redirect_url = 'https://' + data['HOST']
        access_token = ''
        access_token_timestamp = 0.0
        access_token_expires_in = 0
        refresh_token_expires_in = 0
        refresh_token = ''
        res = ''
        if path.exists(access_token_file):
            tokens = Td.load_json_file(access_token_file)
            access_token = tokens['access_token']
            refresh_token = tokens['refresh_token']
            access_token_timestamp = tokens['access_token_timestamp']
            access_token_expires_in = tokens['expires_in']
            refresh_token_expires_in = tokens['refresh_token_expires_in']
        else:
            try:
                res = Td.authentication(client_id, redirect_url, user, secrete)
            except BaseException:
                print("Error happened during dump the access file or authentication")

            #print("authenticated result", res)
            access_token = res['access_token']
            refresh_token = res['refresh_token']
            #print(time.time())
            res['access_token_timestamp'] = time.time()
            access_token_timestamp = res['access_token_timestamp']
            Td.dump_json_file(res, Td.get_access_file_name())
            access_token_expires_in = res['expires_in']
            refresh_token_expires_in = res['refresh_token_expires_in']

        td = Td(access_token, refresh_token, client_id, data['API_KEY'], access_token_timestamp, access_token_expires_in, refresh_token_expires_in)
        td.check_access_token_valid();
        return td
# okay decompiling tda_api.pyc
