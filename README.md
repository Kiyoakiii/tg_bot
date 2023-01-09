# Автоматическая система торговли криптовалютой

## Краткая суть проекта 
Реализовать алгоритм с ИИ торговли на Binance + для удобства вывод в тг информации о сделках и прочие приколы.

![image](https://user-images.githubusercontent.com/113302248/211331783-14212b2f-33bd-4aa3-95be-9a0a84dd8d18.png)

### Get data
#### Training 

У Binance есть API(со своей документацией). Было решено использовать данные начиная с "16 Jun,2022". 

```python
import pandas as pd
from binance.client import Client
import datetime as dt

api_key = ''
api_secret = ''
client = Client(api_key, api_secret)
symbol = "BTCUSDT"
interval='1m'
Client.KLINE_INTERVAL_15MINUTE 
klines = client.get_historical_klines(symbol, interval, "16 Jun,2022")
data = pd.DataFrame(klines)

data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']

data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
data.to_csv(symbol+'_2022_Jun.csv', index = None, header=True)
```
#### Realtime data
