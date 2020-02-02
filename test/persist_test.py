import sys

import numpy as np

sys.path.append("..")
from trade_api.tda_api import Td
from trade_api.mongo_api import mongo_api
from util import load_watch_lists
import os as os
import sys as sys
import json
import getopt
DOWNLOAD_DIR = "/mnt/data_optionistics/data"

def check_num_rows(date_range, symbols, request_type, missing_map):
    collection_name = "optionstat"
    col_names = ["symbol", "date"]
    sort_spec = {"date": 1, "symbol":1}
    if request_type == 'sq':
        collection_name = "stockcandles"

    start_dt, end_dt = date_range.split(",")
    all_hours = Td.get_market_hour(start_dt, end_dt)
    dest_dates = set(all_hours["market_day"])
    m = mongo_api()
    df_all = m.read_df(collection_name, False, col_names, [], {}, sort_spec)
    for d in dest_dates:
        cur_frame = df_all.loc[df_all.date == d]
        diff_symbols = set(symbols) - set(cur_frame.symbol)
        formattedD = d.strftime("%Y%m%d")
        if len(diff_symbols) == 0:
            continue
        if missing_map != {} and d in missing_map.keys():
            missing_map[formattedD].append(diff_symbols)
        else:
            missing_map[formattedD] = list(diff_symbols)
    if missing_map != {}:
        f = open("missing_map_" + request_type + ".json", "w+")
        j = json.dumps(missing_map,  sort_keys=True,indent=4)
        f.write(j)
        f.close()
    else:
        print("foound nothing missing")

def compose_optionistics_file(symbol, date):
    filename = symbol.lower() + ".options." + date + "." + date[4:] + ".csv"
    return filename

def produce_download_scripts(missing_map, request_type):
    download_script_file = open("download_script.sh", "w+")
    persist_script_file = open("persist_script.sh", "w+")
    for d in missing_map.keys():
        symbols = missing_map[d]
        if len(symbols) == 0:
            continue
        if not os.path.exists(DOWNLOAD_DIR + os.sep + d):
            download_script_file.write("python download_option.py -s \"" + str(symbols) + "\" -d " + d + "," + d  + " -t " + request_type + "\n")
        else:
            #there is some data already downloaded, but not inserted
            persist_symbols = []
            for s in symbols:
                fn = compose_optionistics_file(s, d)
                if not os.path.exists(DOWNLOAD_DIR + os.sep + d + os.sep + fn):
                    download_script_file.write("python download_option.py -s " + s + " -d " + d + "," + d + " -t " + request_type + "\n")
                elif request_type == "oq":
                    persist_symbols.append(s)
            if len(persist_symbols) > 0:
                persist_script_file.write("python option_persist.py -d " + DOWNLOAD_DIR + " -s " + str(persist_symbols) + " -p " + d + "\n")

    download_script_file.close()
    persist_script_file.close()

def main(argv):
    symbol = None
    date_range = None
    watch_list = None
    request_type = None
    symbols = []
    skipped_symbols = np.array(["VIAB", "DIA", "HPE", "SPY","QQQ", "DOW","USO","PYPL","IWM", "DD","ARNC", "UAA", "KHC", "BBT", "CBS", "COTY"])
    try:
        opts, args = getopt.getopt(argv, "hs:d:w:t:",
                                   ["help", "symbol=",  "date_range=", "type="])
        for opt, arg in opts:
            if opt == '-h':
                print(sys.argv[0] + '-s <symbol> -d <date_range>  -w <watch_list> -t <oq|sq>')
                sys.exit()
            elif opt in ("-s", "--symbol"):
                symbol = arg
                symbols = [symbol]
            elif opt in ("-d", "--date_range"):
                date_range = arg
            elif opt in ("-w", "--watchlist"):
                watch_list = arg
                symbols = load_watch_lists(watch_list)
            elif opt in ("-t", "--type"):
                request_type = arg
    except getopt.GetoptError:
        print('predictive_model.py -s symbolName -d <date_range, can be 1, 2, 3, 4, 5 year> -c<passcode> -o<output> -w<watchlistfile>')
        sys.exit(2)
    symbols_diff = np.setdiff1d(symbols, skipped_symbols)
    symbols = symbols_diff
    missing_map = {}
    check_num_rows(date_range, symbols, request_type, missing_map)
    if missing_map != {} and request_type == 'oq':
        produce_download_scripts(missing_map, request_type)
if __name__ == "__main__":
    main(sys.argv[1:])
