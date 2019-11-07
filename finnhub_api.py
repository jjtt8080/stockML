# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\Umeda\python\stockML\finnhub_api.py
# Size of source mod 2**32: 1485 bytes
import requests
from datetime import date
from datetime import time
DEBUG = 4
INFO = 2
NONE = 0

class Finnapi:
    api_key = ''
    curr_symbol = ''
    request_string = ''
    response = ''
    log_level = NONE

    def __init__(self, key, loglevel):
        self.api_key = key
        self.log_level = loglevel

    def log(self, cur_loglevel, msg):
        if cur_loglevel <= self.log_level:
            print(msg)

    def call_api(self, api_action, params):
        self.log(DEBUG, 'call_api')
        self.request_string = 'https://finnhub.io/api/v1/stock/' + api_action + '?symbol=' + self.curr_symbol + '&'
        for k in params.keys():
            self.request_string += k
            self.request_string += '='
            self.request_string += params[k]
            self.request_string += '&'

        self.request_string += 'token='
        self.request_string += self.api_key
        self.log(DEBUG, self.request_string)
        self.response = requests.get(self.request_string)
        res = self.response.json()
        return res

    def getOptionChain(self):
        self.log(INFO, 'Getting option chain for symbol ' + self.curr_symbol)
        params = {}
        self.log(DEBUG, params.keys())
        res = self.call_api('option-chain', params)
        return res

    def getCandleStick(self, symbol):
        response = call_api('candle', symbol, {'resolution': '10'})
# okay decompiling finnhub_api.pyc
