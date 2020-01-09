import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt, json
import os as os;
import shutil as shutil;
from datetime import datetime
import zipfile


def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 


def load_json_file(f):
    with open(f, 'r') as (wFile):
        data = json.load(wFile)
        return data


def delta_days(x):
    d1 = x[" DataDate"]
    d2 = x["Expiration"]
    d1 = datetime.strptime(d1, "%m/%d/%Y")
    d2 = datetime.strptime(d2, "%m/%d/%Y")
    return abs((d2 - d1).days)


def load_watch_lists(watch_list_file):
    if (os.path.exists(watch_list_file) == False):
        print("watch_list_file", watch_list_file)
        exit(2, "file does not exist", watch_list_file)
    watch_lists = load_json_file(watch_list_file)
    final_list = []
    for w in watch_lists["watch_lists"]:
        curr_watch = load_json_file(os.path.dirname(watch_list_file) + os.sep + w + ".json")
        final_list = Union(final_list, curr_watch[w])
    return final_list;


def prune_option_file(f, all_quotes_set):
    df = pd.read_csv(f)
    row_before = df.shape[0]
    col_before = df.shape[1]
    mask = df[df["UnderlyingSymbol"].isin(all_quotes_set)]
    mask = mask[(mask["Volume"] > 0) | (mask["OpenInterest"] > 0)]
    if "AKA" in mask.columns:
        mask = mask.drop(["AKA"], axis=1)
    if "days_to_expire" not in mask.columns:
        mask["days_to_expire"] = mask.apply(delta_days, axis=1)
    if " DataDate" in mask.columns:
        mask = mask.drop([" DataDate"], axis=1)
    if "Exchange" in mask.columns:
        mask = mask.drop(["Exchange"], axis=1)
    row_after = mask.shape[0]
    col_after = mask.shape[1]
    print("file", f, row_before, " ", row_after)
    if row_after < row_before or col_after != col_before:
        mask.to_csv(f, index=False)
        shutil.move(f, os.path.dirname(f) + os.sep + "processed" + os.sep)
    return row_before - row_after;
    #else:
    #    return 0;


def prune_stats_file(f, all_quotes_set):
    df = pd.read_csv(f)
    row_before = df.shape[0]
    col_before = df.shape[1]
    mask = df[df["symbol"].isin(all_quotes_set)]
    mask = mask.drop(["quotedate"], axis=1)
    row_after = mask.shape[0]
    col_after = mask.shape[1]
    print("file", f, row_before, " ", row_after)
    if row_after < row_before or col_after < col_before:
        mask.to_csv(f, index=False)
        shutil.move(f, os.path.dirname(f) + os.sep + "processed" + os.sep)
        return row_before - row_after;
    else:
        return 0;
    
    
def prune_files(watch_list_file, datadir):
    if (os.path.exists(datadir) == False):
        exit(2, "Directory does not exist", datadir)
   
    all_quotes = load_watch_lists(watch_list_file) 
    all_quotes_set = set(all_quotes)
    print("datadir", datadir, all_quotes_set)
    total_rows_reduced = 0;
    total_file_processed = 0;
    for (root,dirs,files) in os.walk(datadir, topdown=True): 
        for f in files:
            original_f = f;
            f = datadir + os.sep + f;
            print(f)
            if (original_f[0:8] == 'options_'):
                total_rows_reduced += prune_option_file(f, all_quotes_set)
                total_file_processed += 1;
            elif (original_f[0:11] == 'optionstats') or (original_f[0:11] == 'stockquotes'):
                total_rows_reduced += prune_stats_file(f, all_quotes_set)
                total_file_processed += 1;
        break;
    print("total_rows_reduced ", total_rows_reduced, ", total_file_processed:", total_file_processed)


def process_files(watch_list_file, datadir, prefix):
    zipdir = datadir + os.sep + "zipfiles"
    if (os.path.exists(zipdir) == False):
        exit(2, "Directory does not exist", zipdir)
    for (root, dirs, files) in os.walk(zipdir, topdown=True):
        dest_dir = datadir
        for f in files:
            if not f.endswith('.zip'):
                continue
            if prefix != "" and f.startswith(prefix) == False:
                continue
            real_file_name = zipdir + os.sep + f
            try:
                with zipfile.ZipFile(real_file_name, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
                    prune_files(watch_list_file, dest_dir)
            except zipfile.BadZipFile:
                print("Bad zip file found", real_file_name)
                continue

        break

    
def main(argv):
    symbol = None
    data_dir = None
    watch_list_file = None
    prefix = ""
    
    try:
        opts, args = getopt.getopt(argv, "hd:w:p:", ["help", "data="])
    except getopt.GetoptError:
      print('filter_historical.py -d datadir -w watchlist_file -prefix')
      sys.exit(2)
                                                   
    for opt, arg in opts:
        if opt == '-h':
            print('filter_historical.py -d <datadir> -w <watchlist_file> -p <prefix of zipfiles>')
            sys.exit()
        elif opt in ("-d", "--data"):
            data_dir = arg     
        elif opt in ("-w", "--watchlist"):
            watch_list_file = arg
        elif opt in ("-p", "--prefix"):
            prefix = arg
                                
    process_files(watch_list_file, data_dir, prefix);


if __name__ == "__main__":
    main(sys.argv[1:])