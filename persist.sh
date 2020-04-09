#!/bin/bash
source /home/jane/anaconda3/bin/activate
cd /home/jane/python/tradeML
for i in 20150102  20191209 20191118  20200102 20191203 20200109 20191204  20200110
do 
   echo "persisting ~/Downloads/data/$i"
   python option_persist.py  -d "/home/jane/Downloads/data/$i" -w data/highvol_watchlist.json >> "logs/persist.log_$i" 2>&1 
done
