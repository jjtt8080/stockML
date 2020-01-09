import json
import os as os

DEBUG_LEVEL = 1
import pandas as pd
import numpy as np
from trade_api.tda_api import Td
def debug_print(*argv):
    if DEBUG_LEVEL==1:
        for arg in argv:
            print(arg)

def append_df(df_out, df):
    if df_out is None:
        df_out = df
    else:
        df_out = df_out.append(df, ignore_index=True)
    return df_out

def get_daily_symbols(watch_file_list):
    for f in watch_file_list:
        w = Td.load_json_file(f)


def load_watch_lists(watch_list_file):
    if not os.path.exists(watch_list_file):
        print ("file does not exist", watch_list_file)
        exit(2)

    watch_lists = load_json_file(watch_list_file)
    final_list = []
    for w in watch_lists["watch_lists"]:
        curr_watch = load_json_file(os.path.dirname(watch_list_file) + os.sep + w + ".json")
        final_list = Union(final_list, curr_watch[w])
    return final_list


def load_json_file(f):
    with open(f, 'r') as (wFile):
        data = json.load(wFile)
        return data


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def delta_days(d1, d2):
    return abs((d2 - d1).days)