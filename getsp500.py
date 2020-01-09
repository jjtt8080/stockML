import bs4 as bs
import datetime as dt
import os
import pickle
import requests
import sys
import json
g_sectors_wiki_page = 'https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard'
g_sp500_wiki_page = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

g_dowjones_wiki_page = 'https://money.cnn.com/data/dow30/'
g_nasdaq_wiki_page = 'https://en.wikipedia.org/wiki/NASDAQ-100'

def get_tableElementsFromWiki(pageURL, tableType, colIndex, name, categories=None, categoryIndex=None, categoryName=None):
    resp = requests.get(pageURL)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': tableType})
    result = set()
    categoryMap = {}
    for row in table.findAll('tr')[1:]:
        c = row.findAll('td')[colIndex].text
        if (name == 'dowjones'):
            i = row.findAll('td')[colIndex]
            c = i.findAll('a')[0].text
        c = c.strip('\r')
        c = c.strip('\n')
        result.add(c)
        if categories != None and categoryIndex != None:
            cat = row.findAll('td')[categoryIndex].text
            cat = cat.strip('\r')
            cat = cat.strip('\n')
            if cat in categoryMap.keys():
                categoryMap[cat].append(c);
            else:
                categoryMap[cat] = [c]

    result = list(set(result));
    
    if len(categoryMap) > 0:
        with open(name + "_" + categoryName + ".pickle", "wb") as f:
            pickle.dump(categoryMap, f)
            f.close();
        with open(name + "_" + categoryName + ".json", "w") as f:
            json.dump(categoryMap, f);
            
    
    with open(name + ".pickle", "wb") as f:
        pickle.dump(result, f)
        f.close();

    with open(name + ".json", "w") as f:
        namedObj = {name : result}
        json.dump(namedObj, f);
        f.close();
    
    return (result, categoryMap);
    
def get_sp500sectors():
    (sectors, _) = get_tableElementsFromWiki(g_sectors_wiki_page, 'wikitable', 1,'sectors');
    return sectors;
        
def get_sp500_tickers_by_sectors(sectors):
    (sp500tickers, sectorsMap) = get_tableElementsFromWiki(g_sp500_wiki_page, 'wikitable sortable', 0, 'sp500', sectors, 3, 'sectors');

    return (sp500tickers, sectorsMap)

def get_dows_tickers():
    dowjones_tickers = get_tableElementsFromWiki(g_dowjones_wiki_page, "wsod_dataTable wsod_dataTableBig", 0, 'dowjones');
    
    return dowjones_tickers;

def get_nasdaq_tickers():
    nasdaq_tickers = get_tableElementsFromWiki(g_nasdaq_wiki_page, 'wikitable sortable', 1, 'nasdaq100');
    return nasdaq_tickers;


def main(argv):
    sectors = get_sp500sectors()
    (tickers, sector_tickers) = get_sp500_tickers_by_sectors(sectors)
    
    for s in sector_tickers.keys():
        print("sector ", s, "has " , len(sector_tickers[s]) , ' tickers');

    dow_tickers = get_dows_tickers();
    nasdaq_tickers = get_nasdaq_tickers();
    
if __name__ == "__main__":
    main(sys.argv[1:])
