import datetime
import sys

import numpy as np


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
