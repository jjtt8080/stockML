import datetime
import sys
import numpy as np
def main(argv):
    print(datetime.datetime.fromtimestamp(np.int(np.int(argv[0]))/1000))


if __name__ == "__main__":
    main(sys.argv[1:])
