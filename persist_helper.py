import datetime
import os
import random
import sys

from util import get_stockcandles_for_day, post_processing_for_close

pickles_dir = 'stock_pickles'
def main(argv):
    d1 = argv[0]
    d2 = argv[1]
    watch_list_file = argv[2]
    symbols = []
    if argv[3] is not None:
        symbol = argv[3]
        symbols = [symbol]
        print("symbols", symbols)
    if not os.path.exists(watch_list_file):
        watch_list_file = None
    df = get_stockcandles_for_day(d1, d2, watch_list_file, symbols)
    #df = pd.read_pickle('stock.pickle2020-01-2122')
    df_date = datetime.datetime.strptime(d2,"%Y%m%d")
    df_out = post_processing_for_close(df, df_date)
    temp = str(random.randint(51, 100))
    file_suffix = df_date.strftime("%Y-%m-%d")
    df_out.to_pickle( pickles_dir + os.sep + "stock.pickle" + file_suffix + "_" + temp)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("error in arguments, except  <d1  fmat %Y%m%d> <d2  fmat %Y%m%d> <watchlistfilename> <symbol>")
        exit(1)
    main(sys.argv[1:])
