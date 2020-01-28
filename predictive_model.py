import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys as sys
import os as os
import getopt as getopt
from sklearn.preprocessing import MinMaxScaler, Normalizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from trade_api.tda_api import Td as Td
from optionML import getResultByType
import datetime
from monthdelta import monthdelta
from trade_api.mongo_api import mongo_api
import timedelta
symbol = None
request_type = ''
parameters = ''
code = ''

td = None
opts = None
outputF = None
outputD = None
watch_list_file = None
sc = MinMaxScaler(feature_range=(0, 1))
sc_vol = MinMaxScaler(feature_range=(0, 1))
sc_put_ratio = MinMaxScaler(feature_range=(0, 1))
sc_callvol_ratio =  MinMaxScaler(feature_range=(0, 1))
sc_putvol_ratio =  MinMaxScaler(feature_range=(0, 1))
fmt = "%Y-%m-%d"
DEBUG_LEVEL = 1

## Define some model parameters
STEP_SIZE = 60
predicted_range = 5
NUM_FEATURES = 4
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
    return df


def preprocess_files(symbol, code, begin, end):
    dataset = getPHFromDb(symbol, begin, end)
    optionHist = getOptionHist(symbol, begin, end)
    if optionHist is None or optionHist.shape[0] == 0:
        optionHist = computeOptionHist(symbol, begin, end)
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

    t_set = dataset.loc[:, "close"].values
    t_set_vol = dataset.loc[:,"volume"].values
    o_set = optionHist["put_ratio"].values
    o_set = o_set.reshape(-1,1)
    o_set_vol = optionHist["median_iv_call"].values
    o_set_vol = o_set_vol.reshape(-1,1)
    t_set = t_set.reshape(-1, 1)
    t_set_scaled = sc.fit_transform(t_set)
    t_set_vol = t_set_vol.reshape(-1,1)
    t_set_vol_scaled = sc_vol.fit_transform(t_set_vol)
    o_set_scaled = sc_put_ratio.fit_transform(o_set)
    o_vol_scaled = sc_callvol_ratio.fit_transform(o_set_vol)
    X = []
    Y = []
    X_unscaled = []
    Y_unscaled = []
    steps = 0
    for i in range(STEP_SIZE, t_set.shape[0]-predicted_range):
        for j in range(i-STEP_SIZE, i):
            if NUM_FEATURES == 4:
                X.append((t_set_scaled[j,0],  t_set_vol_scaled[j, 0], o_set_scaled[j,0], o_vol_scaled[j,0]))
            elif NUM_FEATURES == 1:
                X.append(t_set_scaled[j, 0])
            else:
                print("error # of feature")
                exit(1)
            X_unscaled.append((t_set[j,0], t_set_vol[j,0]))
        Y.append(t_set_scaled[i:i+predicted_range, 0])
        Y_unscaled.append(t_set[i:predicted_range,0])
        steps +=1
    X_t, Y_t = np.array(X), np.array(Y)
    X_t = X_t.reshape(steps, STEP_SIZE, NUM_FEATURES)
    print(X_unscaled[-STEP_SIZE:-1])
    print(Y_unscaled[-3:])
    print(len(Y))
    return(X_t, Y_t, X_unscaled, Y_unscaled)


def model(X_train, y_train):
    regressor = Sequential()
    regressor.add(LSTM(units=STEP_SIZE, return_sequences=True, input_shape=(X_train.shape[1], NUM_FEATURES)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=40, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=40))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=predicted_range))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=20, batch_size=16)
    return regressor;


def predict(model,X_test):
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(predicted_stock_price)
    return predicted_stock_price


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


def main(argv):
    symbol = None
    date_range = (500,60)
    output_filename = None
    watch_list = None
    output_model = None
    my_modele  = None

    try:
        opts, args = getopt.getopt(argv, "hs:d:c:o:w:f:",
                                   ["help", "symbol=",  "date_range=", "code=", "output=", "watchlist=", "features_num="])
        for opt, arg in opts:
            if opt == '-h':
                print(sys.argv[0] + '-s <symbol> -d <date_range> -c <code>  -o output -w <watch_list> -f <NUM_FEATURES>')
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
    except getopt.GetoptError:
        print('predictive_model.py -s symbolName -d <date_range, can be 1, 2, 3, 4, 5 year> -c<passcode> -o<output> -w<watchlistfile>')
        sys.exit(2)
    (train_range, test_range) = date_range.split(',')
    print("tran_range, test_range=",  train_range, test_range)
    train_begin, train_end = getDateByCount(train_range)
    test_begin, test_end = getDateByCount(test_range)
    (X_train, Y_train, X_train_unscaled, Y_train_unscaled)  = preprocess_files(symbol, code,  train_begin, train_end)
    (X_test, Y_test, X_test_unscaled, Y_test_unscaled) = preprocess_files(symbol, code, test_begin, test_end)

    if output_filename is not None and os.path.exists(output_filename+".json"):
        my_model = load(output_filename + ".json", output_filename + ".h5")
    else:
        output_filename = "temp_" + datetime.datetime.strftime(datetime.datetime.today(), '%y%m%d')
        my_model = model(X_train, Y_train)
        save(my_model, output_filename)
    if len(X_test) <= 0:
        print("Not enough testing data")
        exit(1)
    predicted = predict(my_model, X_test)


if __name__ == "__main__":
    main(sys.argv[1:])
