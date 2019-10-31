from ib_insync import *
from ib_insync.contract import *
from ib_insync.ticker import Ticker
from time import sleep, strftime


class ibapi:
    m_host = '127.0.0.1'
    m_port = 5000
    m_clientid = 999
    m_ib = None
    m_symbol = ''
    m_contract = None

    def __init__(self, host, port, clientid):
        self.m_host = host
        self.m_port = port
        self.m_clientid = clientid

    def connect(self):
        self.m_ib = IB()
        print("connecting to", self.m_host, self.m_port)
        if (not self.m_ib.isConnected()):
            self.m_ib.connect(self.m_host, self.m_port, 999)

    def disconnect(self):
        self.m_ib.disconnect()

    def setSymbol(self, symbol):
        self.m_symbol = symbol
        self.m_contract = Stock(self.m_symbol, 'NASDAQ', 'USD', includeExpired=False)

    def reqHist(self, c_list, duration, stepSize):
        b_list = []
        for c in c_list:
            bar = self.m_ib.reqHistoricalData(c, '', duration, stepSize, "MIDPOINT", False)
            b_list.append(bar)
        return b_list

    def reqTickers(self, cList):
        priceList = self.m_ib.reqTickers(*cList)
        return priceList

    def reqMktPrice(self, c, genericTickList):
        t = self.m_ib.reqMktData(c, genericTickList, True, False, None)
        return t

    def qualifyContracts(self, symbolList):
        contractList = []
        for s in symbolList:
            print("symbol", s)
            c = Stock(s, 'SMART', 'USD')
            contractList.append(c)
        contractList = self.m_ib.qualifyContracts(*contractList)
        return contractList

    def reqOptionDetails(self, c_list):
        t = self.m_ib.reqMktData(c_list[0], "101", False, False)
        print('len(ib.pendingTickers()):', len(self.m_ib.pendingTickers()))
        print(t.dict())
        self.m_ib.sleep(5)

        print('OI-1 conId', t.contract.conId, 'localSymbol:', t.contract.localSymbol, ',callOI:', t.callOpenInterest,
                  ',callVolumn:', t.volume, "last:", t.last, ",ask:", t.ask, ",bid:", t.bid, ",askSize:",t.askSize)

    def reqOptionChain(self, symbol, exp, strikes, optionType):
        c = Option(symbol, exp, strikes, optionType, 'SMART', 100, 'USD')
        c = self.m_ib.qualifyContracts(c)
        return c

