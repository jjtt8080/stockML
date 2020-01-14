#!/bin/bash
datestr=$(date '+%Y-%m-%d')
source /home/jane/anaconda3/bin/activate
cd /home/jane/python/tradeML
for i in 2014 2015 2016 2017 2018 2019
do 
   filename="highvol_$i.pickle"
   echo "getting watch list $filename"
   python analyze_historical.py  -d data/historical -w data/highvol_watchlist.json -p "$i" -o "$filename" -a DETAIL:monthly
done
