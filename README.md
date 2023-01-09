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

Необходимо получить индикаторы из того, что есть. 

Какие индикаторы используем?

Вот такие:

```python
# Commodity Channel Index 
def CCI(data, ndays): 
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()), name = 'CCI') 
    data = data.join(CCI) 
    return data

# Ease of Movement 
def EVM(data, ndays): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data 

# Simple Moving Average 
def SMA(data, ndays): 
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA') 
    data = data.join(SMA) 
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                    name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA) 
    return data

def BBANDS(data, window):
    MA = data.Close.rolling(window).mean()
    SD = data.Close.rolling(window).std()
    data['UpperBB'] = MA + (2 * SD) 
    data['LowerBB'] = MA - (2 * SD)
    return data

# Force Index 
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data 

# Rate of Change (ROC)
def ROC(data,n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D,name='Rate of Change')
    data = data.join(ROC)
    return data 
```

Почему их? Потому что каждый из них выглядит логично. По сути это комбинации стратегий торговли. В файле тренировки модели есть графики с этими индикаторами. 

#### Нормализация данных. 

Нормализуем данные. Это нужно для того чтобы модель могла понять, что делать если вдруг цена на входе упадет на 100 и станет равная условно 14900, в тренировачном наборе данных такого может не быть поэтому номарлизуем чтоб предотвратить это. Этот  

```prolog
d = preprocessing.normalize(X_train, axis=0)
X_train = pd.DataFrame(d, columns=X_train.columns)
```

#### Выбор библиотеки

Сначла я пытался все делать в tf, потом в  pytorch. Поиспользовав то и то выбор неожиданно пал на pytorch lightning. 
C Pytorch lightning цикл обучения мне стал более понятным + всякие приколы с графиками + lightning_logs + checkpoint_callback, все так просто. 

Каких-то недостатков я не заметил. Кроме того что в интернете по этой библиотеке меньше примеров. В целом, юзабельно.

#### Постановка задачи 

Регрессия или мультиклассификация или бинарная или что.

Тут все хаотично.

Этап 1. 

Так как сначало планировалось решить соревнование с kaggle https://www.kaggle.com/competitions/g-research-crypto-forecasting, то и проекте планировалось предсказывать лог доходность, а занчит и регрессию . Позже стало ясно, что не понятно где использовать ее в алгоритме торговли. Тут можно прикрутить её юзабельность, но есть и по лучше идеи.  

Этап 2.

Бинарная классификация. Предсказывать вход в потанциадльную сделку. Тут я решил использовать CrossEntropyLoss(), почему не BCEWithLogitsLoss()? 

Этап 3.

Потому что я занал что хочу проверить паттерн нейтрального класса в данной задаче. Он не сработал. Модель не улучшилась, а бинарная классификация то, что нам нужно. 

Итог: хотим посторить модель, которая предсказывает потенциально выиграшный вход в сделку. 

ФОТО

#### Выбор модели

До этого проекта я не работал с временными рядами, а в процессеобучения в deep learning school от МФТИ + mlcourse.ai не уделял должного внимания временным рядам. Но я всегда знал про lstm и что они хорошо справляются с времеными рядами. 

! Важно отметить, что не получилось релаизовать в данном разделе. Я подумал, что круто было бы реализвоать LSTM + ССА или для предобработки последовательности использовать гауссовские процессы. Но потратив недельку понял, что  времени не хватит. 
Поэтому ограничились LSTM. 




