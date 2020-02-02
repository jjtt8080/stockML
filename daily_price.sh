#!/bin/bash
for i in Default sp500 dowjones nasdaq100
do 
   filename="data/$i.json"
   echo "getting watch list $filename"
   python optionML.py -w "$filename" -c 2048 -t price_history -o data
done
