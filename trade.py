'''
TRADE
'''

# 
#import the libraries
import pandas as pd
#download tick data for AAPL stock
data = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=^BVSP&interval=1min&outputsize=full&apikey=UDCCZOE6N1IMGNER&datatype=csv")
data

#https://www.alphavantage.co/documentation/


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
df.head()

trace = go.Candlestick(x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])
data = [trace]
py.iplot(data, filename='simple_candlestick')