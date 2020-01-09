import numpy as np
import pandas as pd
import sys, getopt, json
import os as os;
from datetime import datetime
import tarfile
import calendar

calendar.setfirstweekday(6)

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def load_json_file(f):
    with open(f, 'r') as (wFile):
        data = json.load(wFile)
        return data


def delta_days(d1, d2):
    return abs((d2 - d1).days)


class OptionStats:
    d_begin = None
    d_end = None
    filename = ""
    weekday_df = None
    monthly_df = None
    year_df = None
    raw_df = None
    weekday_stock_df = None
    monthly_stock_df = None
    year_stock_df = None
    stock_raw_df_map = {}
    stock_raw_df = None
    opdetail_raw_df = None
    all_quotes = None

    def __init__(self, date_begin, date_end, raw_df_path, quotes):
        # Create a empty data frame for all months
        # We will accumulate the metrix in  weekday level, monthly level, yearly level, quarter level,
        self.d_begin = datetime.strptime(date_begin, "%Y%m%d")
        self.d_end = datetime.strptime(date_end, "%Y%m%d")
        self.all_quotes = quotes
        if os.path.exists(raw_df_path) and os.path.isfile(raw_df_path):
            self.raw_df = pd.read_pickle(raw_df_path)
            self.stock_raw_df = pd.read_pickle(raw_df_path + '_stock')
            print("loaded", self.raw_df.columns)
            print("loaded", self.stock_raw_df.columns)

    @staticmethod
    def append_to_raw_df(df_all, df):
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.append(df, ignore_index=True)
        return df_all

    @staticmethod
    def append_to_symbol_raw_df(dfmap, s, df):
        if s in dfmap.keys():
            dfmap[s] = dfmap[s].append(df[df["symbol"] == s], ignore_index=True)
        else:
            dfmap[s] = pd.DataFrame(df[df["symbol"]==s])
        return dfmap

    @staticmethod
    def calculate_intrinsic_value(x):
        strike = x["Strike"]
        if (x["Type"] == 'call' and x["Strike"] > x["UnderlyingPrice"]) or (x["Type"] == 'put' and x["Strike"] < x["UnderlyingPrice"]):
            intrinsic_value = 0
        else:
            if x["UnderlyingPrice"] is None:
                print("X strange", x)
            intrinsic_value = np.abs(x["Strike"] - x["UnderlyingPrice"])
        return intrinsic_value

    @staticmethod
    def calculate_time_value(x):
        try:
            mid_price = (x["Bid"] + x["Ask"]) / 2
            time_value = mid_price - x["intrinsic_value"]
            return time_value
        except TypeError:
            print("x", x)
            exit(-1)

    @staticmethod
    def extract_expiration_date(x):
        x["exp_date"] = pd.to_datetime(x["Expiration"], format="%m/%d/%Y")
        return x

    @staticmethod
    def week_number_of_month(date_value):
        return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)

    def add_detail_day(self,d, df, analyze_option):
        d_cur = datetime.strptime(d, "%Y%m%d")
        df = df.drop(['OptionExt','Gamma','Vega'], axis=1)
        df = df[df["UnderlyingSymbol"].isin(self.all_quotes)]
        df = df[(df["Delta"] > 0.4) | (df["Delta"] > -0.3)]
        df = df[(df["days_to_expire"] < 90)]
        df = OptionStats.extract_expiration_date(df)
        df["data_date"] = d_cur
        df["exp_year"], df["exp_month"], df["exp_week"], df["exp_day"] = \
            df.exp_date.apply(lambda x: x.year), \
            df.exp_date.apply(lambda x: x.month), \
            df.exp_date.apply(lambda x: OptionStats.week_number_of_month(x)), \
            df.exp_date.apply(lambda x: x.day)

        if analyze_option.find("monthly") > 0:
            df = df[df["exp_week"] == 3]
        df["intrinsic_value"] = df.apply(lambda x: OptionStats.calculate_intrinsic_value(x), axis=1)
        df["time_value"] = df.apply(lambda x: OptionStats.calculate_time_value(x), axis=1)
        self.opdetail_raw_df = self.append_to_raw_df(self.opdetail_raw_df, df)
        return self.opdetail_raw_df.shape[0]

    @staticmethod
    def calculate_true_range(df):
        df = df.sort_values(by=['symbol', 'year', 'month', 'd_index'])
        df["prev_close"] = df["close"].shift()
        df["range_1"] = df["high"] - df["low"]
        df["range_2"] = df["high"] - df["prev_close"]
        df["range_3"] = df["prev_close"] - df["low"]
        df["tr"] = df[["range_3","range_2"]].apply(lambda x: max(x["range_3"], x["range_2"]), axis=1)
        df["tr"] = df[["tr", "range_1"]].apply(lambda x: max(x["tr"], x["range_1"]), axis=1)
        index_of_high = list(df.columns).index("high")
        index_of_low = list(df.columns).index("low")
        index_of_tr = list(df.columns).index("tr")
        df.iloc[0, index_of_tr] = df.iloc[0, index_of_high] - df.iloc[0, index_of_low]
        df = df.drop("range_1", axis=1)
        df = df.drop("range_2", axis=1)
        df = df.drop("range_3", axis=1)
        df = df.drop("prev_close", axis=1)
        return df

    def add_stats_day(self, d, df, dftype):
        d_cur = datetime.strptime(d, "%Y%m%d")
        d_index = delta_days(d_cur, self.d_begin)
        df["d_index"] = d_cur.day
        df["day_index"] = d_cur.isoweekday()
        df["month"] = d_cur.month
        df["year"] = d_cur.year
        if dftype == 'optionstats':
            self.raw_df = self.append_to_raw_df(self.raw_df, df)
        elif dftype == 'stockquotes':
            for s in df["symbol"]:
                self.stock_raw_df_map = OptionStats.append_to_symbol_raw_df(self.stock_raw_df_map, s, df)

    @staticmethod
    def aggregate_optionstats(df):
        return df.agg(
            calliv_min=('calliv', np.min),calliv_max=('calliv', np.max), calliv_mean=('calliv', np.mean), calliv_std=('calliv', np.std),
            putiv_min=('putiv', np.min), putiv_max=('putiv', np.max), putiv_mean=('putiv', np.mean),putiv_std=('putiv', np.std),
            callvol = ('callvol', np.min), callvol_max = ('callvol', np.max), callvol_mean = ('callvol',np.mean), callvol_std = ('callvol', np.std),
            putvol=('putvol', np.min), putvol_max=('putvol', np.max), putvol_mean=('putvol', np.mean),
            putvol_std=('putvol', np.std),
            calloi=('calloi', np.min), calloi_max=('calloi', np.max), calloi_mean=('calloi', np.mean),
            calloi_std=('calloi', np.std),
            putoi=('putoi', np.min), putoi_max=('putoi', np.max), putoi_mean=('putoi', np.mean),
            putoi_std=('putoi', np.std),
        )

    def aggregate_stockstats(self, df):
        return df.agg(
            price_mean=('close', np.mean),  atr=('tr', np.mean),\
            volume_min=('volume', np.min), volume_max=('volume', np.max), \
            volume_mean=('volume', np.mean),volume_std=('volume', np.std)
        )

    @staticmethod
    def aggregate_columns(self, df, func):
        try:
            weekday_df = func(df.groupby(by=["symbol", "day_index"], as_index=True))
            print(weekday_df)
            monthly_df = func(df.groupby(by=["symbol", "month"], as_index=True))
            print(monthly_df)
            year_df = func(df.groupby(by=["symbol", "year"], as_index=True))
            print(year_df)
            return weekday_df, monthly_df, year_df
        except KeyError as e:
            print("Key error({0})".format(e))
            print(df.columns)

    def start_analyze(self, analyze_options):
        if self.raw_df is not None:
            print("raw_df shape", self.raw_df.shape)
            print(self.raw_df.describe(include=[np.number]))
        if self.stock_raw_df_map is not {} :
            print("stock raw map size", len(self.stock_raw_df_map.keys()))
            for s in self.stock_raw_df_map.keys():
                print("examing for particular symbol", s)
                self.stock_raw_df_map[s] = OptionStats.calculate_true_range(self.stock_raw_df_map[s])
        if analyze_options == 'ALL':
            print("aggregating...")
            (self.weekday_df, self.monthly_df, self.year_df) = self.aggregate_columns(self.raw_df, self.aggregate_optionstats)
            (self.weekday_stock_df, self.monthly_stock_df, self.year_stock_df) = \
                self.aggregate_columns(self.stock_raw_df, self.aggregate_stockstats)

    def save(self, filename, analyze_options):
        print("saving ...")
        if analyze_options.startswith('RAW'):
            if self.raw_df is not None:
                self.raw_df.to_pickle(filename)
            if self.stock_raw_df_map is not {}:
                for s in self.stock_raw_df_map.keys():
                    print("appending ", self.stock_raw_df_map[s].shape)
                    self.stock_raw_df = self.append_to_raw_df(self.stock_raw_df, self.stock_raw_df_map[s])
                self.stock_raw_df.to_pickle(filename + "_stock")

        if analyze_options.startswith('DETAIL'):
            self.opdetail_raw_df.to_pickle(filename + "_detail")

        if analyze_options == 'ALL':
            self.monthly_df.to_pickle(filename+"_monthly")
            self.weekday_df.to_pickle(filename+"_weekday")
            self.year_df.to_pickle(filename + "_yearly")
            self.weekday_stock_df.to_pickle(filename+"_stock_weekday")
            self.monthly_stock_df.to_pickle(filename+"_stock_monthly")
            self.year_stock_df.to_pickle(filename+"_stock_yearly")


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


def append_option_detail(f, optionstats, analyze_option):
    df = pd.read_csv(f)
    filename = os.path.basename(f)
    dateStr = filename[8:16] # due to the file naming convension
    return optionstats.add_detail_day(dateStr, df, analyze_option)


def append_stats_file(f, optionstats, stats_type):
        df = pd.read_csv(f)
        print(df.shape[0])
        filename = os.path.basename(f)
        dateStr = filename[12:20]
        optionstats.add_stats_day(dateStr, df, stats_type)
        return df.shape[0]


def preprocess_files(all_quotes, opstats, datadir, prefix, outputfile, analyze_options):
    if not os.path.exists(datadir):
        print("directory does not exist", datadir)
        exit(2)
    
    all_quotes_set = set(all_quotes)
    print("datadir", datadir, all_quotes_set)
    total_rows_analyzed = 0
    total_file_processed = 0
    print("starting , prefix =", prefix, datadir)
    for (root,dirs,files) in os.walk(datadir, topdown=True):
        for f in files:
            original_f = f

            f = datadir + os.sep + f

            if analyze_options.startswith('DETAIL') and original_f[0:8] == 'options_':
                if prefix != '' and original_f[8:12] != prefix:
                    continue
                print(f)
                total_rows_analyzed = append_option_detail(f, opstats, analyze_options)
                total_file_processed += 1
            elif analyze_options.startswith('RAW'):
                if prefix != '' and original_f[12:16] != prefix:
                    continue
                print(f)
                if (analyze_options.endswith('optionstats') and original_f.startswith('optionstats')) or \
                    (analyze_options.endswith('stockquotes') and original_f.startswith('stockquotes')) or \
                    analyze_options =='RAW':
                    total_rows_analyzed += append_stats_file(f, opstats, original_f[0:11])
                    total_file_processed += 1
                    #os.remove(f)
            if total_file_processed % 100 == 0:
                print("total file processed, total row finished", total_file_processed, total_rows_analyzed)
        break
    print("total_rows_analyzed ", total_rows_analyzed, ", total_file_processed:", total_file_processed)


def process_files(watch_list_file, datadir, prefix, output_filename, analyze_options):
    zipdir = datadir + os.sep + "tarfiles"
    all_quotes = load_watch_lists(watch_list_file) 
    if analyze_options in ['RAW'] and not os.path.exists(zipdir):
        print("Directory does not exist", zipdir)
        exit(2)

    dest_dir = datadir + os.sep + 'analyzed'
    pickles_dir = datadir + os.sep + 'pickles'
    print("analyze_options", analyze_options, "zipdir", zipdir)
    optionstats = None
    if analyze_options.startswith('RAW'):
        optionstats = OptionStats("20000101", "20191231", output_filename, all_quotes)
        for (root, dirs, files) in os.walk(zipdir, topdown=True):
            for f in files:
                print(f)
                if not f.endswith('.tar'):
                    continue;
                if prefix != "" and not f.startswith(prefix):
                    continue;

                print("extracting", f)
                with tarfile.open(zipdir + os.sep + f, 'r') as tar_ref:
                    tar_ref.extractall(dest_dir)
                    preprocess_files(all_quotes, optionstats, dest_dir, prefix, output_filename, analyze_options)
            break
    elif analyze_options.startswith('DETAIL'):
        optionstats = OptionStats("20000101", "20191231", pickles_dir + os.sep + output_filename, all_quotes)
        preprocess_files(all_quotes, optionstats, dest_dir, prefix, output_filename, analyze_options)
    elif analyze_options.startswith('ALL'):
        optionstats = OptionStats("20000101", "20191231", pickles_dir + os.sep +  output_filename, all_quotes)

    if optionstats is not None:
        optionstats.start_analyze(analyze_options)
        optionstats.save(pickles_dir + os.sep +  output_filename, analyze_options)

    
def main(argv):
    symbol = None
    data_dir = None
    watch_list_file = None
    prefix = ""
    output_filename = ""
    analyze_options = ""
    try:
        opts, args = getopt.getopt(argv, "hd:w:p:o:a:", ["help", "data=", "prefix=", "output=", "analyze="])
    except getopt.GetoptError:
        print(sys.argv[0] + ' -d datadir -w watchlist_file -p prefix -o output -a <RAW,DETAIL,ALL>')
        sys.exit(2)
                                                   
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0] + '-d <datadir> -w <watchlist_file> -p <prefix> -o output -a analyze')
            sys.exit()
        elif opt in ("-d", "--data"):
            data_dir = arg     
        elif opt in ("-w", "--watchlist"):
            watch_list_file = arg
        elif opt in ("-p", "--prefix"):
            prefix = arg
        elif opt in ("-o", "--output"):
            output_filename = arg
        elif opt in ("-a", "--analyze"):
            analyze_options = arg
                                
    process_files(watch_list_file, data_dir, prefix, output_filename, analyze_options);
if __name__ == "__main__":
    main(sys.argv[1:])
