import os
import datetime
import timedelta

for (root,dirs,files) in os.walk('data', topdown=True):
    for f in files:
        if not f.endswith(".csv"):
            continue
        tokens = f.split("_")
        datestr = "20" + tokens[0]
        dateObj = datetime.datetime.strptime(datestr,'%Y%m%d')
        if dateObj >= datetime.datetime.strptime("20191220", '%Y%m%d') and dateObj <= datetime.datetime.strptime("20200104", '%Y%m%d'):
            newObj = dateObj - datetime.timedelta(days=1)
            dest = datetime.datetime.strftime(newObj, '%Y%m%d')[2:]
            dest += "_"
            dest += tokens[1]
            dest += "_"
            dest +=  tokens[2]
            print("renaming " + root + os.sep + f + " to " + root + os.sep + dest)