import datetime
import os
import random
import sys

from util import get_stockcandles_for_day, post_processing_for_close,load_watch_lists, post_process_ph_df
from trade_api.db_api import construct_day_filter
from trade_api.mongo_api import mongo_api
collection_name = 'stockcandles'
pickles_dir = 'stock_pickles'


def find_missing_symbols(target_symbols, target_date):
    m = mongo_api()
    date_filter = construct_day_filter(target_date)
    symbol_in_db = m.read_df(collection_name, False, ["symbol", "date"], [], date_filter, {})
    if symbol_in_db.shape[0] > 0:
        symbols_missing = set(target_symbols) - set(symbol_in_db["symbol"].values)
        print("symbols_in_db", symbol_in_db.shape, symbols_missing)
        return list(symbols_missing)
    else:
        return target_symbols
def main(argv):
    d1 = argv[0]
    d2 = argv[1]
    watch_list_file = argv[2]
    symbols = []
    if not os.path.exists(watch_list_file) :
        symbol = argv[2]
        symbols = [symbol]
        print("symbols", symbols)
    else:
        symbols = load_watch_lists(watch_list_file)
    if not os.path.exists(watch_list_file):
        watch_list_file = None
    df_date = datetime.datetime.strptime(d2, "%Y%m%d")
    symbols = find_missing_symbols(symbols, df_date)
    print("symbols after calculating missing", symbols)
    df = get_stockcandles_for_day(d1, d2, None, symbols, False)

    #df = pd.read_pickle('stock.pickle2020-01-2122')
    print("df.shape after get_stockcandles_for_day", df.shape)
    df_out = post_processing_for_close(df, df_date)
    temp = str(random.randint(51, 100))
    file_suffix = df_date.strftime("%Y-%m-%d")
    filename = pickles_dir + os.sep + "stock.pickle" + file_suffix + "_" + temp
    df_out.to_pickle(filename)
    if df[df_out.chg.isna()].shape[0] == 0:
        print("persisting... shape is:", df.shape)
        m = mongo_api()
        m.write_df(df_out, 'stockcandles')
    print("saving to " + filename)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("error in arguments, except  <d1  fmat %Y%m%d> <d2  fmat %Y%m%d> <watchlistfilename> <symbol>")
        exit(1)
    main(sys.argv[1:])
