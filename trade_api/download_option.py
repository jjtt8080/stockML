import datetime
import getopt
import json
import os as os
import os.path
import pickle
import sys
import time
from shutil import which

import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import Select

sys.path.append("..")
from trade_api.tda_api import Td

loginurl = "https://www.optionistics.com/secure/ssllogin.pl"
url = "http://www.optionistics.com/c/download.pl"
download_dir = "/home/jane/python/tradeML/data_optionistics/"
options = webdriver.ChromeOptions()
chrome_driver_binary = which('chromedriver') or '/usr/local/bin/chromedriver'

username=""
password=""
driver = webdriver.Chrome(chrome_driver_binary, chrome_options=options)
cookie_file = '/tmp/cookie'

def save_cookie(driver, path):
    with open(path, 'wb') as filehandler:
        pickle.dump(driver.get_cookies(), filehandler)


def load_cookie(driver, path):
    with open(path, 'rb') as cookiesfile:
        cookies = pickle.load(cookiesfile)
        for cookie in cookies:
            if 'expiry' in cookie:
                del cookie['expiry']
            print("add cookie", cookie)
            driver.add_cookie(cookie)

def login(url, username, password):
    driver.get(url)
    user = username or os.environ.get('OPTION_USER', '')
    password = password or os.environ.get('OPTION_PASS', '')

    if user:
        ubox = driver.find_element_by_id('uname')
        pbox = driver.find_element_by_id('pwd')
        ubox.send_keys(user)
        pbox.send_keys(password)
        driver.find_element_by_name('remember').click()
        driver.find_element_by_name('li').click()
        save_cookie(driver, cookie_file)
        time.sleep(2)

def selectElementByValue(driver, elementName, elementValue, byid=True):
    if byid:
        select = Select(driver.find_element_by_id(elementName))
    else:
        select = Select(driver.find_element_by_name(elementName))
    select.select_by_value(elementValue)

def download_symbol(url, symbol, start_month, start_day, end_month, end_day, year, type):
    symbol = symbol.lower()
    desc =  ".options."
    if type == 'sq':
        desc = ".stock."
    date_desc = year + start_month.zfill(2) + start_day.zfill(2)
    filename =  symbol + desc + date_desc + "." + end_month.zfill(2) + end_day.zfill(2) + ".csv"
    file_path = download_dir + filename
    file2_path = download_dir + "data" + os.sep + filename
    print("file_path", file_path)
    if os.path.exists(file_path) or os.path.exists(file2_path):
        print("file exists", file_path)
        return

    try:
        driver.find_element_by_name('symbol').clear()
        driver.find_element_by_name('symbol').send_keys(symbol)
        selectElementByValue(driver, 'type', type, False)
        selectElementByValue(driver, 'stmth', start_month)
        selectElementByValue(driver, 'stday', start_day)
        selectElementByValue(driver, 'endmth', end_month)
        selectElementByValue(driver, 'endday', end_day)
        selectElementByValue(driver, 'year', year)
        if not driver.find_element_by_name('tos').is_selected():
            driver.find_element_by_name('tos').click()
        driver.find_element_by_name('go').click()
        driver.find_element_by_name('dl').click()
    except selenium.common.exceptions.UnexpectedAlertPresentException:
        print("Error ")
    sleep_timer = 0
    while not os.path.exists(file_path):
        time.sleep(1)
        sleep_timer += 1
        if sleep_timer >= 7:
            break


def download(url, watch_list_file, start_month,start_day,end_month, end_day,year, type):
    load_cookie(driver, cookie_file)
    print("load url", url)
    driver.get(url)
    if watch_list_file is not None:
        watch_lists = Td.load_json_file(watch_list_file)
        symbols = watch_lists["high_vol"]
    for symbol in symbols:
        print("getting symbol", symbol)
        try:
            download_symbol(url, symbol, start_month,start_day,end_month, end_day,year, type)
        except:
            print("Error getting symbo.", symbol)
            continue
    driver.close()
    driver.quit()

def load_json_for_symbol(symbol):
    symbols = []
    if symbol.find("'") != -1:
        symbol = symbol.replace("'", "\"")
    symbol = '{\"symbols\":' + symbol + '}'
    if symbol is not None:
        symbols = json.loads(symbol)
        return symbols["symbols"]
    return None

def main(argv):
    symbol = None
    start_date = None
    end_date = None
    watch_list = None
    type = "oq"
    try:
        opts, args = getopt.getopt(argv, "h:s:d:w:t:", ["help",  "symbol=", "date_range=", "watch_list=", "type="])
    except getopt.GetoptError:
        print(sys.argv[0] + '-s symbol -d date_range')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + ' -s symbol -d date_range <yyyymmdd,yyyymmdd>')
            sys.exit()
        elif opt == '-d':
            date_range = arg
            print(date_range)
            (start_date, end_date) = date_range.split(',')
            start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
            end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        elif opt == '-s':
            symbol = arg

        elif opt == '-w':
            watch_list = arg
        elif opt == '-t':
            type = arg
    if (symbol is None and watch_list is None) or start_date is None or end_date is None:
        exit(1)

    login(loginurl, username, password)
    print("start_date.month, start_date.day", start_date.month, start_date.day, end_date.month, end_date.day)
    symbols = []
    if watch_list is None and symbol.find("[") != -1:
        symbols = load_json_for_symbol(symbol)
        print("symbols", symbols)
        if len(symbols) > 0:
            load_cookie(driver, cookie_file)
            print("load url", url)
            driver.get(url)
            for s in symbols:
                print("getting symbols", s)
                download_symbol(url, s, str(start_date.month), str(start_date.day), str(end_date.month), str(end_date.day), str(start_date.year), type)
    elif symbol is not None:
        load_cookie(driver, cookie_file)
        driver.get(url)
        print("load url", url)
        download_symbol(url, symbol, str(start_date.month), str(start_date.day), str(end_date.month), str(end_date.day), str(start_date.year), type)
    if watch_list is not None:
        download(url, watch_list, str(start_date.month), str(start_date.day), str(end_date.month), str(end_date.day), str(start_date.year), type)

if __name__ == "__main__":
        main(sys.argv[1:])
