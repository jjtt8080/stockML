from ib_api import ibapi
from ib_insync import *
from tabulate import tabulate
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
ib_a = ibapi('127.0.0.1', 4001, 999)
ib_a.connect()
SYMBOL_LIST = ['FB', 'AAPL']
c_list = ib_a.qualifyContracts(SYMBOL_LIST)
p_list = ib_a.reqTickers(c_list)


for p in p_list:
    print(p.marketPrice())
c = ib_a.reqOptionChain('XOP', '20191220', 23, 'C')
p = ib_a.reqOptionDetails(c)
#print("all keys", t.dict().values())
#print("Option volume", t.dict().get('volume'))
#print("Option last ", t.dict().get('last'))
#print("Option Interest", t.dict().get('callOpenInterest'))