import pandas as pd
from binance.client import Client
import datetime as dt


api_key = 'gD1bn84kRIxFXaB09Q7faOaxScvcWcxw3hXvAEKrmVGFYrmu7IRpS48GkWP8ZFWi'
api_secret = 'lCYT2ngPiaG5tJdpHBZPfAN2mo7xJq2mvVMti03zrRZFQbOsdgLhc9KPlDBPfAwW'
client = Client(api_key, api_secret)

symbol = "BTCUSDT"
interval='1m'
Client.KLINE_INTERVAL_15MINUTE 
klines = client.get_historical_klines(symbol, interval, "16 Jun,2022")
data = pd.DataFrame(klines)

data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']

data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
data.to_csv(symbol+'_2022_Jun.csv', index = None, header=True)

df=data.astype(float)
df["close"].plot(title = 'DOTUSDT', legend = 'close')
