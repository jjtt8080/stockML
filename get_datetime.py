import datetime
import sys
import dateutil
import numpy as np

def gettimestampForPeriod(d1, period='D', count=-10):
    if period == 'D':
        delta = dateutil.relativedelta.relativedelta(days=count)
    elif period == 'M':
        delta = dateutil.relativedelta.relativedelta(months=count)
    elif period == 'Y':
        delta = dateutil.relativedelta.relativedelta(years=count)
    #else:
    #    raise ValueError("unknown parameter", period, count)

    d2 = d1 + delta
    print("d1, d2", d1, d2)
    d1_stmp = datetime.datetime.strftime(d1, '%s')
    d2_stmp = datetime.datetime.strftime(d2, '%s')
    return d1_stmp, d2_stmp

def main(argv):
    if type(argv[0]) == str and argv[0].find(",") == -1:
        print(datetime.datetime.fromtimestamp(np.int(np.int(argv[0]))/1000))
    else:
        list_dates = argv[0].split(",")
        for x in list_dates:
            d_str = datetime.datetime.strptime(x, "%Y%m%d")
            d = d_str.strftime("%s")
            print(d)



if __name__ == "__main__":
    main(sys.argv[1:])
