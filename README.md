# Автоматическая система торговли криптовалютой

## Краткая суть проекта 
Реализовать алгоритм с ИИ торговли на Binance + для удобства вывод в тг информации о сделках и прочие приколы.

![image](https://user-images.githubusercontent.com/113302248/211331783-14212b2f-33bd-4aa3-95be-9a0a84dd8d18.png)

### Get data
#### Training 

У Binance есть API(со своей документацией). Было решено использовать данные начиная с "16 Jun,2022". Нам нужны только ```
['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']``` поэтому остальное дропаем. 

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

Это нам понадобится, для того чтобы предсказывать в релаьном времени точку входа в сделку. 
```python
cl = Spot()
r = cl.klines('BTCUSDT', '1m', limit =  500)
df = DataFrame(r).iloc[:, :9]
df = df.drop([6, 7], axis = 1)
df.columns = ('timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Count')
```
Тут ничего интересного. Можно отметить только момент с ```limit```, он не больше 1000. То есть, если длина последовательности для LSTM больше 1000, то надо придумывать махинации, а учитывая что индикаторы 'съедают'(например индикаторы, которым надо много исторических данных, ну или nan на нули менять или по умному заполнять, но кажется это лажа и проще просто откинуть строки с nan). Ограничений на количество запросов на скачивание я не получал так что пока опустим момент с ограничениями. 

### Preprocessing data
После того как мы получили данные из прошлого пункта нам необходимо проверить их на пробелы и наны. 
```(df_train.index[1:]-df_train.index[:-1]).value_counts()``` эта штука проверяет на наличие пробелов. Почему они есть?
Вообще говоря зависит от даты с которой скачать тренировочные данные. Если качать с июля 2022 то их не будет, но если скачать с 
июля 2021 они будут. 
```data = data.reindex(range(data.index[0],data.index[-1]+60,60),method='pad')``` - это решает проблему. 
Также сосдаем колонку 'Target' который определяем так:

```python
for i in range (len(df['Open'])):
    open_price= df['Open'][i]
    max = df['High'][i:i+15:].max()
    if max >= open_price + 25:
        df['Target'][i] = 1
    else: 
        df['Target'][i] = 0
```
Логика такая. Если в близжайшие 15 минут есть high_price > 25, в этом случае мы бы заработали денег. Почему + 25?
Во-первых потому что сейчас даже + 5 за 250 минут встречается редко а значит выход из модели будет равен 0(не входим  сделку), и так далее. Почему не + 5? Тогда таргетов с 1 >> чем с 0. А значит нужно думать над перебалансировкой. Так и было, я думал что в данном проекте данный паттерн реализуем, но чисто по-человечески: если модель предсказывает + 5 то разве это то что хотя бы в теории может заработать денег? Так что было принято решение оставить + 25. 

Дальше по файлу всякие типичные штучки и ничего интересного. 

### Processing data


