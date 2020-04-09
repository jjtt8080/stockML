short_date_str=`echo $(date '+%Y%m%d')  | cut -c3-`
. /home/jane/anaconda3/bin/activate
cd /home/jane/python/tradeML
python stock_persist.py data/highvol_watchlist.json > dailyoutput/stockpersist_${short_date_str}.log 2>&1

sleep 50s

python option_persist.py -d data -w data/highvol_watchlist.json -p ${short_date_str} > dailyoutput/optionpersist_${short_date_str}.log 2>&1 

