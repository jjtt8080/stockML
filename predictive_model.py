import datetime
import getopt as getopt
import os as os
import sys as sys
import time
import pandas as pd
import numpy as np
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from monthdelta import monthdelta
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import gc
from optionML import getResultByType
from trade_api.mongo_api import mongo_api
from trade_api.tda_api import Td
from trade_api.ta_api import get_features
from util import set_random_seeds, load_watch_lists, drop_columns

symbol = None
request_type = ''
parameters = ''
code = ''

td = None
opts = None
outputF = None
outputD = None
watch_list_file = None

fmt = "%Y-%m-%d"
DEBUG_LEVEL = 1


# Define some model parameters
STEP_SIZE = 60



option_feature_names = ["put_ratio", "median_iv_call", "mean_vol_call", "max_vol_call", "max_vol_put"]
scaler_map = {}
predict_files_dir = "predictions"


def getScalerByFeatureName(feature_name, feature_range_low=0, feature_range_high=1):
    if feature_name in scaler_map.keys():
        return scaler_map[feature_name]
    else:
        #sc = MinMaxScaler(feature_range=(feature_range_low, feature_range_high))
        sc = StandardScaler()
        scaler_map[feature_name] = sc
        return sc
def filter_base_on_option(ta_options, column_names, stock_feature_names):
    fil_obj = filter(lambda x: x.startswith(ta_options + "_"), list(column_names))
    for f in fil_obj:
        stock_feature_names.append(f)
        print("adding features", f)

def append_ta_stock_feature_names(stock_feature_names, ta_options, df):
    df = df.dropna(axis=1, how='any')
    column_names = df.columns
    if ta_options != 'all':
        filter_base_on_option(ta_options, column_names, stock_feature_names)
    else:
        filter_base_on_option('momentum', column_names, stock_feature_names)
        filter_base_on_option('trend', column_names, stock_feature_names)
        filter_base_on_option('volatility', column_names, stock_feature_names)
        filter_base_on_option('others', column_names, stock_feature_names)
        filter_base_on_option('volume', column_names, stock_feature_names)

    return stock_feature_names

def debug_print(*argv):
    if DEBUG_LEVEL==1:
        for arg in argv:
            print(arg)


def getTrainParameters(range):
    return "{\"resolution\": \"D\", \"count\":" + range  + "}"


def getTestParameters(range):
    return "{\"resolution\": \"D\", \"count\":" + range + "}"


def getDateByCount(day_range):
    end_date = datetime.datetime.today() - datetime.timedelta(days=1)
    if day_range.endswith("D"):
        day_range = day_range[0:len(day_range) - 1]
        date = datetime.datetime.today() - datetime.timedelta(days=np.int(day_range))
        debug_print("resolving date from ", day_range, "to", date.strftime('%Y-%m-%d'))
        return date, end_date

    elif day_range.endswith("M"):
        day_range = day_range[0:len(day_range) - 1]
        date = datetime.datetime.today() - monthdelta(np.int(day_range))
        debug_print("resolving date from ", day_range, "to", date.strftime('%Y-%m-%d'))
        return date, end_date

    elif day_range.endswith("Y"):
        day_range = day_range[0:len(day_range) - 1]
        if day_range.isint():
            day_range = 12 * np.int(day_range)
            date = datetime.datetime.today() - monthdelta(day_range)
            debug_print("resolving date from ", day_range, "to", date.strftime('%Y-%m-%d'))
            return date, end_date

    else:
        begin,end = day_range.split(':')
        begin_date = datetime.datetime.strptime(begin, '%Y%m%d')
        end_date = datetime.datetime.strptime(end, '%Y%m%d')
        return begin_date, end_date

def resetHour(x):
    y = datetime.datetime(x.year, x.month, x.day, 0,0,0)
    return y


def convertDate(t):
    ## For some reason FIN API's stock date is 1 day ahead
    dateS = datetime.datetime.fromtimestamp(float(t)) + datetime.timedelta(days=1)
    dateS = resetHour(dateS)
    return dateS


def computeOptionHist(symbol, startDate, endDate):
    projectionAttrs = ['data_date', 'UnderlyingSymbol']
    projectionMeasures = ['Volume', 'OpenInterest']
    sortSpec = ['UnderlyingSymbol', 'data_date']
    filter =  {'$and': [ {'UnderlyingSymbol': {'$in': [symbol]}},{'Type': {'$eq': 'call'}}, {'data_date': {'$gte': startDate}}, \
                         {'data_date': {'$lte': endDate}}]}
    sortSpec = {'symbol': 1, 'data_date': 1}
    m = mongo_api()
    df_call = m.read_df('optionhist', False, projectionAttrs, projectionMeasures, filter, sortSpec)
    filter = {'$and': [{'UnderlyingSymbol': {'$in': [symbol]}}, {'Type': {'$eq': 'put'}}, {'data_date': {'$gte': startDate}}, \
                         {'data_date': {'$lt': endDate}}]}
    df_put = m.read_df('optionhist', False, projectionAttrs, projectionMeasures, filter, sortSpec)
    df_call.columns = ['date', 'symbol', 'callvol', 'calloi']
    df_put.columns = ['date', 'symbol', 'putvol', 'putoi']
    df = df_call
    df["putvol"] = df_put.putvol
    df["putoi"] = df_put.putoi
    debug_print("after computing optionstat history", df.shape, df.columns)
    m.write_df(df, 'optionstat')
    return df


def getOptionHist(symbol, startDate, endDate):
    projectionAttrs = ["date", 'max_vol_call', 'max_vol_put', 'mean_vol_call', 'mean_vol_put', 'mean_oi_call', 'mean_oi_put', 'median_iv_call']
    projectionMeasures = []
    filter = {'symbol': {'$eq': symbol}}
    sortSpec = {'symbol': 1, 'date': 1}
    m = mongo_api()
    df = m.read_df('optionstat', False, projectionAttrs, projectionMeasures, filter, sortSpec)
    if df is not None and df.shape[0] > 0:
        print("df.columns", df.columns)
        df = df[(df.date >= startDate) & (df.date <= endDate)]
        print("OptionHistory shape", df.shape, df.columns)
    else:
        print("can't find the stats: ", filter, projectionAttrs)
    return df


def compare_dif(stockDf, optionDf):
    stockDf["date"] = stockDf.date.apply(lambda x: resetHour(x))
    optionDf["date"] = optionDf.date.apply(lambda x: resetHour(x))
    stock_date = set(stockDf.date)
    option_date = set(optionDf.date)
    diff2 = stock_date - option_date
    print(diff2)
    return list(diff2)


def getPHromAPI(code, symbol, parameters):
    dataset = getResultByType('price_history', code, symbol, parameters)
    debug_print("stock dataset shape", dataset.shape)
    dataset['data_date'] = dataset.t.apply(lambda x: convertDate(x))
    debug_print(dataset.iloc[0,:])
    debug_print(dataset.iloc[-1,:])
    startDate = np.min(dataset['data_date'])
    maxLen = dataset.shape[0]
    endDate =  np.max(dataset['data_date'])
    print("getting option history from ", startDate, " to ", endDate)


def getPHFromDb(symbol, begin, end):
    m = mongo_api()
    filter = {'$and': [{'symbol':{'$eq': symbol}}, {'date': {'$gte': begin}}, {'date': {'$lt': end}}]}
    print(filter)
    df = m.read_df('stockcandles', False, '*', [], {'$and': [{'symbol':{'$eq': symbol}}, {'date': {'$gte': begin}},{'date': {'$lt': end}}]}, {})
    df = drop_columns(df, ["_id"])
    return df


def scale_single_feature(f, dataset):
    print("fitting", f)
    f_set = dataset.loc[:, f].values
    f_set = f_set.reshape(-1, 1)
    scaler = getScalerByFeatureName(f)
    fset_scaled = scaler.fit_transform(f_set)
    return (fset_scaled, f_set)


def scale_features(dataset, optionHist, stock_feature_names, option_feature_names):
    scaled_stock_features = {}
    scaled_option_features = {}
    unscaled_features = {}
    for f in stock_feature_names:
        scaled_stock_features[f], unscaled_features[f] = scale_single_feature(f, dataset)
    if optionHist is not None and optionHist.shape[0] >0:
        for o in option_feature_names:
            scaled_option_features[o], unscaled_features[o] = scale_single_feature(o, optionHist)
    return scaled_stock_features, scaled_option_features, unscaled_features


def preprocess_files(symbol, code, begin, end, ta_options, predicted_range, NUM_FEATURES):

    dataset = getPHFromDb(symbol, begin, end)
    optionHist = getOptionHist(symbol, begin, end)
    if NUM_FEATURES == 4 and (optionHist is None or optionHist.shape[0] == 0):
        print("change feature for 1 for this symbol, option dataset is not enabled")
        exit(1)
        #optionHist = computeOptionHist(symbol, begin, end)
    if optionHist is not None and optionHist.shape[0] != 0:
        optionHist["put_ratio"] = optionHist["max_vol_put"] / optionHist["max_vol_call"]
        optionHist["callvoi"] = optionHist["mean_vol_call"] / optionHist["mean_oi_call"]
        diff = compare_dif( dataset, optionHist)
        if len(diff) > 1:
            print("insufficient data")
            exit(1)
        elif len(diff) == 1:
            x1 = diff[0].to_pydatetime().date()
            x2 = datetime.datetime.today().date()
            if x1 == x2:
                dataset.drop(dataset.tail(1).index, inplace=True)
            else:
                exit(1)
    stock_feature_names = ["close", "volume", "chg", "ho_spread", "cl_spread"]
    print("technical analysis options", ta_options)
    if ta_options is not None:
        dataset = get_features(dataset, ta_options)
        stock_feature_names = append_ta_stock_feature_names(stock_feature_names, ta_options, dataset)
        if optionHist is None or optionHist.shape[0] == 0:
            NUM_FEATURES = len(stock_feature_names)
        else:
            NUM_FEATURES = len(stock_feature_names)  + len(option_feature_names)
        print("stock_feature_names, num features", stock_feature_names, NUM_FEATURES)
    (stock_features, option_features, unscaled_features) = scale_features(dataset, optionHist, stock_feature_names, option_feature_names)
    X = []
    Y = []
    X_unscaled = []
    Y_unscaled = []
    steps = 0
    for i in range(STEP_SIZE, stock_features["close"].shape[0]-predicted_range):
        for j in range(i-STEP_SIZE, i):
            if NUM_FEATURES == 2:
                X.append((stock_features["close"][j, 0], stock_features["volume"][j, 0]))
            elif NUM_FEATURES >= 4:
                for f in stock_feature_names:
                    if np.isnan(stock_features[f]).any():
                        print("wrong na values exists in", f)
                        exit(1)
                    X.append(stock_features[f][j,0])
                if optionHist is not None and optionHist.shape[0] > 0:
                    for o in option_feature_names:
                        X.append(option_features[o][j,0])
            else:
                print("error # of feature")
                exit(1)
            X_unscaled.append((unscaled_features["close"][j,0], unscaled_features["chg"][j,0]))
        Y.append((stock_features["close"][i:i+predicted_range, 0], stock_features["chg"][i:i+predicted_range, 0]))
        Y_unscaled.append(unscaled_features["chg"][i:predicted_range,0])
        steps +=1
    X_t, Y_t = np.array(X), np.array(Y)
    X_t = X_t.reshape(steps, STEP_SIZE, NUM_FEATURES)
    Y_t = Y_t.reshape(steps, 2*predicted_range)
    return(X_t, Y_t, X_unscaled, Y_unscaled, NUM_FEATURES)


def model(X_train, y_train, NUM_FEATURES, predicted_range):
    regressor = Sequential()
    regressor.add(LSTM(units=STEP_SIZE, return_sequences=True, input_shape=(X_train.shape[1], NUM_FEATURES)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=60))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=predicted_range*2))
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    regressor.compile(optimizer='adam',  loss=['mean_absolute_error'])
    regressor.fit(X_train, y_train, validation_split=0.15, epochs=50, batch_size=20, callbacks=[es])
    return regressor


def predict(model,X_test, output_filename, start_dt, end_dt, predicted_range):
    predicted_stock_price_chg = model.predict(X_test)
    col_base_name = ["close", "chg"]
    colnames = []
    for i in list(range(0, predicted_range)):
        for c in col_base_name:
            col_name = c + str(i)
            colnames.append(col_name)
    predicted_df = pd.DataFrame(data=predicted_stock_price_chg, columns=colnames)
    date_range = Td.get_market_hour(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    d_range = date_range["market_day"].apply(lambda x: x.strftime("%s")).values
    chg_scaler = getScalerByFeatureName("chg")
    close_scaler = getScalerByFeatureName("close")
    for c in colnames:
        if c.startswith("chg"):
            predicted_stock_price_chg = chg_scaler.inverse_transform(predicted_df[c].values.reshape(-1,1)).flatten()
            predicted_df[c] = [float(v) for v in predicted_stock_price_chg]
        if c.startswith("close"):
            predicted_stock_price_close = close_scaler.inverse_transform(predicted_df[c].values.reshape(-1,1)).flatten()
            predicted_df[c] = [float(v) for v in predicted_stock_price_close]
    d_range_prediction_range = d_range[len(d_range) - predicted_df.shape[0]:]
    print("length of dates", len(d_range_prediction_range), "length of predictions", len(predicted_stock_price_chg))
    predictedObj = {"start_dt": start_dt.strftime("%Y%m%d"), "end_dt": end_dt.strftime("%Y%m%d"), \
                    "date": list(d_range_prediction_range)}
    for c in colnames:
        predictedObj[c] = list(predicted_df[c])

    #print(json.dumps(predictedObj, indent=4, default=str))
    Td.dump_json_file(predictedObj, output_filename)
    return predicted_df


def save(model, output_filename):
    model_json = model.to_json()
    with open(output_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights( output_filename + ".h5")


def load(model_filename, weight_filename):
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_filename)
    print("Loaded model from disk")
    return loaded_model

def execute(symbol, code, train_begin, train_end, test_begin, test_end, output_filename, ta_options, predicted_range,NUM_FEATURES):
    my_model = None
   
    (X_train, Y_train, X_train_unscaled, Y_train_unscaled, NUM_FEATURES) = preprocess_files(symbol, code, train_begin, train_end, ta_options, predicted_range, NUM_FEATURES)
    (X_test, Y_test, X_test_unscaled, Y_test_unscaled, NUM_FEATURES) = preprocess_files(symbol, code, test_begin, test_end, ta_options, predicted_range, NUM_FEATURES)
    today_str = datetime.datetime.strftime(datetime.datetime.today(), '%y%m%d')
    if output_filename is None:
        output_filename = symbol + "_"
    else:
        output_filename  = output_filename + "_" + symbol
    real_file_name = output_filename + today_str
    if real_file_name is not None and os.path.exists(real_file_name + ".json"):
        my_model = load(real_file_name + ".json", real_file_name + ".h5")
    else:
        print("**** Training model for symbol", symbol)
        my_model = model(X_train, Y_train, NUM_FEATURES, predicted_range)
        save(my_model, real_file_name)
    if len(X_test) <= 0:
        print("Not enough testing data")
        exit(1)
    prediction_file_name = predict_files_dir + os.sep + symbol + "_" + today_str
    print("**** Predicting for symbol", symbol)
    predict(my_model, X_test, prediction_file_name, test_begin, test_end, predicted_range)
    del my_model
    del X_train, Y_train, X_train_unscaled, Y_train_unscaled
    del X_test, Y_test, X_test_unscaled, Y_test_unscaled
    gc.collect()

def exec_main(argv):
    symbol = None
    date_range = (500,60)
    output_filename = None
    watch_list = None
    ta_options = None
    predicted_range = 1
    NUM_FEATURES = 1
    try:
        opts, args = getopt.getopt(argv, "hs:d:c:o:w:f:t:p:",
                                   ["help", "symbol=",  "date_range=", "code=", "output=", "watchlist=", "features_num=", "technical_analysis=", "predicted_range="])
        for opt, arg in opts:
            if opt == '-h':
                print(sys.argv[0] + '-s <symbol> -d <date_range> -c <code>  -o output -w <watch_list> -f <NUM_FEATURES> -t <momentum, volume, others, volatility, trend -p <predicted_range>')
                sys.exit()
            elif opt in ("-s", "--symbol"):
                symbol = arg
            elif opt in ("-d", "--date_range"):
                date_range = arg
            elif opt in ("-c", "--code"):
                code = arg
            elif opt in ("-o", "--output"):
                output_filename = arg
            elif opt in ("-w", "--watchlist"):
                watch_list = arg
            elif opt in ("-f", "--features_num"):
                NUM_FEATURES = np.int(arg)
            elif opt in ("-t", "--technical_analysis"):
                ta_options = arg
            elif opt in ("-p", "--predicted_range"):
                predicted_range = np.int(arg)
    except getopt.GetoptError:
        print('predictive_model.py -s symbolName -d <date_range, can be 1, 2, 3, 4, 5 year> -c<passcode> -o<output> -w<watchlistfile>')
        sys.exit(2)
    (train_range, test_range) = date_range.split(',')
    print("tran_range, test_range=",  train_range, test_range)

    train_begin, train_end = getDateByCount(train_range)
    test_begin, test_end = getDateByCount(test_range)
    if watch_list is None and symbol is not None:
        execute(symbol, code, train_begin, train_end, test_begin, test_end, output_filename, ta_options, predicted_range, NUM_FEATURES)
    elif watch_list is not None:
        symbols = np.sort(load_watch_lists(watch_list))
        for s in symbols:
            try:
                execute(s, code, train_begin, train_end, test_begin, test_end, output_filename, ta_options, predicted_range,NUM_FEATURES)
            except:
                print("error on symbol", symbol, "continuing")
if __name__ == "__main__":
    set_random_seeds()
    exec_main(sys.argv[1:])

