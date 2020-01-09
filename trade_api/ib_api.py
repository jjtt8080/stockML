# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: C:\Users\Umeda\python\stockML\ib_api.py
# Size of source mod 2**32: 3980 bytes
from ib_insync import *
from ib_insync.contract import *
from ib_insync.ticker import Ticker
from time import sleep, strftime
from ib_insync.objects import BarList, BarDataList, RealTimeBarList, AccountValue, PortfolioItem, Position, Fill, Execution, BracketOrder, TradeLogEntry, OrderState, ExecutionFilter, TagValue, PnL, PnLSingle, ContractDetails, ContractDescription, OptionChain, OptionComputation, NewsTick, NewsBulletin, NewsArticle, NewsProvider, HistoricalNews, ScannerSubscription, ScanDataList, HistogramData, PriceIncrement, DepthMktDataDescription
import asyncio, logging, nest_asyncio

class ibapi:
    m_host = '127.0.0.1'
    m_port = 5000
    m_clientid = 999
    m_ib = None
    m_symbol = ''
    m_contract = None
    m_connected = False

    def ibapi(self, host, port, clientid):
        self.__init__(host, port, clientid)

    def __init__(self, host, port, clientid):
        self.m_host = host
        self.m_port = port
        self.m_clientid = clientid
        self.m_ib = IB()
        self.connect()

    def connect(self):
        if not self.m_connected:
            nest_asyncio.apply()
            asyncio.get_event_loop().set_debug(True)
            #util.logToConsole(logging.INFO)
            loop = asyncio.get_event_loop()
            #print('in ib_api.connect')
            if not self.m_ib.isConnected():
                self.m_ib.connect(self.m_host, self.m_port, self.m_clientid)
                m_connected = True
            else:
                m_connected = True

    def disconnect(self):
        self.m_ib.disconnect()

    def setSymbol(self, symbol):
        self.m_symbol = symbol
        self.m_contract = Stock((self.m_symbol), 'NASDAQ', 'USD', includeExpired=False)

    def reqHist(self, c_list, duration, stepSize):
        b_list = []
        for c in c_list:
            bar = self.m_ib.reqHistoricalData(c, '', duration, stepSize, 'MIDPOINT', False)
            b_list.append(bar)

        return b_list

    def reqTickers(self, cList):
        priceList = (self.m_ib.reqTickers)(*cList)
        return priceList

    def reqMktPrice(self, c, genericTickList):
        t = self.m_ib.reqMktData(c, genericTickList, True, False, None)
        return t

    def qualifyContracts(self, symbolList):
        contractList = []
        for s in symbolList:
            #print('symbol', s)
            c = Stock(s, 'SMART', 'USD')
            contractList.append(c)

        contractList = (self.m_ib.qualifyContracts)(*contractList)
        return contractList

    def reqOptionDetails(self, c_list):
        t = self.m_ib.reqMktData(c_list[0], '101', False, False)
        #print('len(ib.pendingTickers()):', len(self.m_ib.pendingTickers()))
        #print(t.dict())
        self.m_ib.sleep(5)
        #print('OI-1 conId', t.contract.conId, 'localSymbol:', t.contract.localSymbol, ',callOI:', t.callOpenInterest, ',callVolumn:', t.volume, 'last:', t.last, ',ask:', t.ask, ',bid:', t.bid, ',askSize:', t.askSize)

    def reqOptionChain(self, symbol, exp, strikes, optionType):
        c = Option(symbol, exp, strikes, optionType, 'SMART', 100, 'USD')
        c = self.m_ib.qualifyContracts(c)
        return c
    # valid scanner code
    # "LOW_OPT_VOL_PUT_CALL_RATIO", "HIGH_OPT_IMP_VOLAT_OVER_HIST", "LOW_OPT_IMP_VOLAT_OVER_HIST", "HIGH_OPT_IMP_VOLAT",
    # "TOP_OPT_IMP_VOLAT_GAIN", "TOP_OPT_IMP_VOLAT_LOSE", "HIGH_OPT_VOLUME_PUT_CALL_RATIO", "LOW_OPT_VOLUME_PUT_CALL_RATIO",
    # "OPT_VOLUME_MOST_ACTIVE", "HOT_BY_OPT_VOLUME", "HIGH_OPT_OPEN_INTEREST_PUT_CALL_RATIO",
    # "LOW_OPT_OPEN_INTEREST_PUT_CALL_RATIO", "TOP_PERC_GAIN", "MOST_ACTIVE", "TOP_PERC_LOSE", "HOT_BY_VOLUME",
    # "TOP_PERC_GAIN", "HOT_BY_PRICE", "TOP_TRADE_COUNT", "TOP_TRADE_RATE", "TOP_PRICE_RANGE", "HOT_BY_PRICE_RANGE",
    # "TOP_VOLUME_RATE", "LOW_OPT_IMP_VOLAT", "OPT_OPEN_INTEREST_MOST_ACTIVE", "NOT_OPEN",
    # "HALTED", "TOP_OPEN_PERC_GAIN", "TOP_OPEN_PERC_LOSE", "HIGH_OPEN_GAP", "LOW_OPEN_GAP", "
    # LOW_OPT_IMP_VOLAT", "TOP_OPT_IMP_VOLAT_GAIN", "TOP_OPT_IMP_VOLAT_LOSE", "HIGH_VS_13W_HL",
    # "LOW_VS_13W_HL", "HIGH_VS_26W_HL", "LOW_VS_26W_HL", "HIGH_VS_52W_HL", "LOW_VS_52W_HL",
    # "HIGH_SYNTH_BID_REV_NAT_YIELD", "LOW_SYNTH_BID_REV_NAT_YIELD"
    def reqScannerOption(self, scanCode='HIGH_OPT_VOLUME_PUT_CALL_RATIO'):
        scanSub = ScannerSubscription()
        scanSub.instrument = 'STK'
        scanSub.locationCode = 'STK.US.MAJOR'
        scanSub.scanCode = scanCode
        tagvalues = []
        tagvalues.append(TagValue('usdMarketCapAbove', '10000'))
        tagvalues.append(TagValue('optVolumeAbove', '1000'))
        tagvalues.append(TagValue('avgVolumeAbove', '5000000'))
        c = self.m_ib.reqScannerData(scanSub, None, tagvalues)
        return c
# okay decompiling ib_api.pyc
