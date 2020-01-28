import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt, json
from getpass import getpass
from trade_api.finnhub_api import Finnapi as Finnapi
from trade_api.tda_api import Td as Td
from trade_api.ib_api import ibapi as ibapi
from cryptography.fernet import Fernet
from datetime import date
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import date as date
# doing some setup
import asyncio
from trade_api.mongo_api import mongo_api

def init_ib():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ib_a = ibapi('127.0.0.1', 4001, 999)
    ib_a.connect()
    return ib_a

def init_token():
    password = b"password"
    # salt = os.urandom(16)
    print("Command prompt key generation for td user/password")
    user = input("Please enter user: ")
    secrete = getpass()
    salt = input("a number to encrpyt your password:")
    td = Td.init(user, secrete)
    salt = salt.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    f = Fernet(key)
    secrete = secrete.encode("utf-8")
    token = f.encrypt(secrete)

    outputJson = {"user": user, "token": token.decode("utf-8")}
    tdconfig_json = Td.get_config_dir() + 'tdconfig.json'
    print(outputJson)
    Td.dump_json_file(outputJson, tdconfig_json)
    print("key generate successfully! file location:" + tdconfig_json)

def init_td(passcode):
    password = b"password"
    #salt = os.urandom(16)
    salt = passcode.encode("utf-8")
    #print("decrpyting from passcode" + passcode)
    config_file = Td.get_config_dir() + 'tdconfig.json'
    data = Td.load_json_file(config_file)
    user = data["user"]
    token = data["token"]
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    f = Fernet(key)
    secrete = f.decrypt(token.encode("utf-8"))
    secrete = secrete.decode("utf-8")
    td = Td.init(user, secrete)
    return td


def getDefaultOptionParameters(symbol, type):
     parameters =  {"symbol": symbol, "contractType": type, "strikeCount": 25,
                    "monthIncrement": 3, "strike": '',
                    "interested_columns":
                        ['expirationDate', 'strikePrice','putCall', 'totalVolume', 'openInterest','last', 'bid', 'ask', 'highPrice', 'lowPrice', 'delta', 'mark', 'netChange', 'theta', 'volatility'],"monthlyOption": False}
     return parameters

def getOptions(parameters, symbol, code):
    td = init_td(code)
    if parameters == '':
        param = getDefaultOptionParameters(symbol, 'ALL')
        print("param", param)
        result = td.get_option_chain(param)
        #param = getDefaultOptionParameters(symbol, 'PUT')
        #resultPut = td.get_option_chain(param)
        #result = resultCall.append(resultPut)
    else:
        result = (td.get_option_chain(parameters))
         
    return result;

def getDefaultPriceHistoryParameters():
    return "{\"periodType\": \"day\",\"frequencyType\": \"minute\", \"period\":1, \"frequency\": 15}"
    

def getDailyPriceHistoryParameters():
    return "{\"periodType\": \"month\",\"frequencyType\": \"daily\", \"period\":1, \"frequency\": 1}"

def generateOutputFileName(outputDir, symbol, requestType):
    todayStr = date.today().strftime('%y%m%d')
    outputFile = outputDir + os.sep + symbol + os.sep + todayStr + "_"  + requestType;
    outputFile += ".csv"
    if os.path.exists(outputDir) == False:
        os.mkdir(outputDir)
    if os.path.exists(outputDir + os.sep + symbol) == False:
        os.mkdir(outputDir + os.sep + symbol)
    return outputFile;
    
def getResultByType(request_type,code,symbol, parameters, use_td=False):
    result = None
    if request_type == 'quote':
        #ib_a = init_ib()
        #c_list = ib_a.qualifyContracts([symbol])
        #p_list = ib_a.reqTickers(c_list)
        #for p in p_list:
        #    print(p.marketPrice())
        td = init_td(code);
        result = td.get_quotes(symbol)
    elif request_type == 'init_token':
        init_token()

    elif request_type == 'option_chain':
        result = getOptions(parameters, symbol, code);

    elif request_type == 'price_history':
        if code is not None or use_td:
            td = init_td(code)
            if parameters == '':
                parameters =  getDefaultPriceHistoryParameters()
            parameters = json.loads(parameters)
            result = td.get_price_history(symbol, None, None, parameters)
        else:
            #print(parameters)
            f = Finnapi('', 2)
            result = f.getCandleStick(symbol, parameters)
    elif request_type == 'option_history':
        m = mongo_api()
        result = m.read_df('optionstats', False, \
                      [{'date': {'$dateFromParts': {'year': '$year', 'month': '$month', 'day': '$d_index'}}}, 'symbol'], \
                      ['callvol', 'calloi', 'putvol', 'putoi'], \
                      {'symbol': {'$in': [symbol]}}, \
                       {'symbol': 1, 'date': 1})

    return result


def getResultByWatchList(watchlist_file, outputDir, request_type, code):
    td = init_td(code)
    w =  Td.load_json_file(watchlist_file)
    error_list = ['NULL']
    final_list = []
    for l in w.keys():
        curr_list = w[l]
        print(curr_list)
        final_list = list(set(curr_list) |  set(final_list))
    if (len(final_list) == 0):
        return None;
    print("get result for the list " , final_list);
    for s in final_list:
        f = generateOutputFileName(outputDir, s, request_type)
        if os.path.exists(f):
            print("existing file", f)
            continue;
        try:
            print("getting option for symbol", s)
            result = getResultByType(request_type, code, s, '')
            if result is not None:
                result.to_csv(f, index=False);
            else:
                print("empty result set for symbol ", s)
        except ValueError as error:
            print("error getting", s, error);
            error_list = error_list.append(s)
            continue;
        except ConnectionError:
            error_list = error_list.append(s);
            continue;
        except KeyError:
            error_list = error_list.append(s);
            continue;
    print(error_list)
    
def main(argv):
    symbol = None
    request_type = ''
    parameters = ''
    code = None

    td = None
    opts = None
    outputF = None
    outputD = None
    watch_list_file = None
    
    try:
        opts, args = getopt.getopt(argv, "hs:t:p:c:o:w:", ["help", "symbol=", "type=", "params=", "code=", "output=", "watchlist="])
    except getopt.GetoptError:
      print('optionML.py -s symbolName -t type  <option_chain,quote,price_history,init_token> -p<options> -c<passcode> -o<output> -w<watchlistfile>')
      sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('optionML.py -s <symbol> -t <option_chain,quote,price_history,init_token> -p<options> -c<passcode>')
            sys.exit()
        elif opt in ("-s", "--symbol"):
            symbol = arg
        elif opt in ("-t", "--type"):
            request_type = arg
        elif opt in ('-p', "--params"):
            parameters = arg
        elif opt in ('-c', "--code"):
            code = arg
        elif opt in ('-o', "--output"):
            if os.path.isdir(arg):
                outputD = arg
            else:
                outputF = arg
        elif opt in ('-w', "--watchlist"):
            if (os.path.isfile(arg) == False) or (os.path.exists(arg) == False):
                exit(2, "Error, watch list file does not exist")
            else:
                watch_list_file = arg
                
    if symbol != None:
        result = getResultByType(request_type,code,symbol, parameters)
        if (outputF != None):
            result.to_csv(outputF, index=False);
        else:
            print(result)
    elif watch_list_file != None and outputD != None:
        getResultByWatchList(watch_list_file, outputD, request_type, code);
                

    else:
        print("error in argument")
        exit(-1)
        
if __name__ == "__main__":
    main(sys.argv[1:])
