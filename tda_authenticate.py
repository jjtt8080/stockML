# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\Umeda\python\stockML\tda_authenticate.py
# Size of source mod 2**32: 4372 bytes
import os, os.path, sys, requests, time
from selenium import webdriver
from shutil import which
from urllib.parse import parse as up
import json

def authentication(client_id, redirect_uri, tdauser=None, tdapass=None):
    client_id = client_id + '@AMER.OAUTHAP'
    print('client_id:', client_id)
    url = 'https://auth.tdameritrade.com/auth?response_type=code&redirect_uri=' + up.quote(redirect_uri) + '&client_id=' + up.quote(client_id)
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
    elif os.path.exists('C:/Program Files/Google/Chrome/Application/chrome.exe'):
        options.binary_location = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
    else:
        chrome_driver_binary = which('chromedriver') or '/usr/local/bin/chromedriver'
        driver = webdriver.Chrome(chrome_driver_binary, chrome_options=options)
        print('found driver', driver)
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
        resp = requests.post('https://api.tdameritrade.com/v1/oauth2/token', headers={'Content-Type': 'application/x-www-form-urlencoded'},
          data={'grant_type':'authorization_code', 
         'refresh_token':'', 
         'access_type':'offline', 
         'code':code, 
         'client_id':client_id, 
         'redirect_uri':redirect_uri})
        assert not resp.status_code != 200, 'Could not authenticate!'
        return resp.json()


def refresh_token(refresh_token, client_id):
    resp = requests.post('https://api.tdameritrade.com/v1/oauth2/token', headers={'Content-Type': 'application/x-www-form-urlencoded'},
      data={'grant_type':'refresh_token', 
     'refresh_token':refresh_token, 
     'client_id':client_id})
    assert not resp.status_code != 200, 'Could not authenticate!'
    return resp.json()


def main():
    config_file = 'config.json'
    data = []
    try:
        with open(config_file) as (json_file):
            data = json.load(json_file)
    except BaseException:
        print("can't open config.json file")

    client_id = data['API_KEY']
    redirect_url = 'https://' + data['HOST']
    print(authentication(client_id, redirect_url))


if __name__ == '__main__':
    main()
# okay decompiling tda_authenticate.pyc
