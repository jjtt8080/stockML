from util import load_watch_lists
import sys
import os
import path
import numpy as np
import subprocess

def main(argv):
    watch_list = "data/highvol_watchlist.json"
    symbols = np.sort(load_watch_lists(watch_list))
    print(argv)
    for s in symbols:
        try:
            newargv = []
            newargv.append('python')
            newargv.append('predictive_model.py')
            for a in argv:
                newargv.append(a)
            newargv.append("-s")
            newargv.append(s)
            print(newargv)
            process = subprocess.Popen(newargv,
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True)
            while True:
                output = process.stdout.readline()
                print(output.strip())
                # Do something else
                return_code = process.poll()
                if return_code is not None:
                    print('RETURN CODE', return_code)
                    # Process has finished, read rest of the output
                    for output in process.stdout.readlines():
                        print(output.strip())
                    break
        except os.error:
            print("error on symbol", s, "continuing")

if __name__ == "__main__":
    main(sys.argv[1:])

