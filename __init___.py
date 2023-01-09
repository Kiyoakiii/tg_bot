from binance.spot import Spot
import pandas as pd 
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

import torch.nn as nn
import torch.optim as optim

from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional import accuracy

from torchmetrics.functional import accuracy
from torch.autograd import Variable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional import accuracy
from torch.autograd import Variable
import pandas as pd

from datetime import datetime
import time

from config import api_key, api_secret


import requests
import telebot
from bs4 import BeautifulSoup as b  
import schedule
import time
import pandas as pd
import numpy as np

import seaborn as sns

from binance.client import Client
import requests
import numpy as np
import talib
import time
import decimal 

token = '' # API
id = ''

start_time = datetime.now()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
print(device)

BATCH_SIZE = 1
buy = False
sell = True
price_buy = 0
SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
LIMIT = '200'
QNTY = decimal.Decimal('0.00355')
client = Client(api_key, api_secret)
predictions =  np.array([0])
bot = telebot.TeleBot(token)
sym = 'btcusdt'



def dataset():
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
    cl = Spot()
    r = cl.klines('BTCUSDT', '1m', limit =  500)
    df = DataFrame(r).iloc[:, :9]
    df = df.drop([6, 7], axis = 1)
    df.columns = ('timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Count')
    data = pd.DataFrame(df)
    data['Open'] = data['Open'].astype(object).astype(float)
    data['High'] = data['High'].astype(object).astype(float)
    data['Low'] = data['Low'].astype(object).astype(float)
    data['Close'] = data['Close'].astype(object).astype(float)
    data['Volume'] = data['Volume'].astype(object).astype(float)
    data['Count'] = data['Count'].astype(object).astype(float)

    BATCH_SIZE = 1
    seq_length = 250
    df = data
    df['Target']= 0

    for i in range (len(df['Open'])):
        open_price= df['Open'][i]
        max = df['High'][i:i+15:].max()
        if max >= open_price + 20:
            df['Target'][i] = 1
        else: 
            df['Target'][i] = 0
    n = 20
    NIFTY_ROC = ROC(df,n)
    ROC = NIFTY_ROC['Rate of Change']
    df = NIFTY_ROC

    n = 50
    NIFTY_BBANDS = BBANDS(df, n)
    pd.concat([NIFTY_BBANDS.Close,NIFTY_BBANDS.UpperBB,NIFTY_BBANDS.LowerBB],axis=1).plot(figsize=(9,5),grid=True)
    df = NIFTY_BBANDS

    n = 20
    AAPL_ForceIndex = ForceIndex(df,n)
    ForceIndex = AAPL_ForceIndex['ForceIndex']
    df = AAPL_ForceIndex

    # Compute the 14-day Ease of Movement for AAPL
    n = 14
    AAPL_EVM = EVM(df, n)
    AAPL_EVM  = AAPL_EVM.dropna()
    EVM = AAPL_EVM['EVM']
    df = AAPL_EVM

    n = 60*3
    NIFTY_CCI = CCI(df, n)
    NIFTY_CCI = NIFTY_CCI.dropna()
    CCI = NIFTY_CCI['CCI']
    df = NIFTY_CCI

    n = 9
    SMA_NIFTY = SMA(df,n)
    SMA_NIFTY = SMA_NIFTY.dropna()
    SMA = SMA_NIFTY['SMA']
    df = SMA_NIFTY

    ew = 15
    EWMA_NIFTY = EWMA(df,ew)
    EWMA_NIFTY = EWMA_NIFTY.dropna()
    EWMA = EWMA_NIFTY['EWMA_'  + str(ew)]
    df = EWMA_NIFTY

    df_train = df
    df_train = df_train.set_index("timestamp")

    target = 'Target'
    drops = ['timestamp']
    features = [f for f in df_train.columns if f not in drops + [target]]

    X_train = df_train[features]
    y_train = df_train[target]

    print(y_train.value_counts())

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train)
    y_train["label"] = encoded_labels

    sequences =[]
    for idx in range(len(X_train) - seq_length):
        x = X_train[idx:idx + seq_length]
        y = y_train[idx:idx + seq_length].iloc[0]
        sequences.append((x, y))
    return sequences, features, label_encoder

test_sequences, features, label_encoder = dataset()

class SurfaceDataset(Dataset):
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence = torch.Tensor(sequence.values),
            label = torch.tensor(label).long()
        )

class SurfaceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = SurfaceDataset(self.train_sequences)
        self.test_dataset = SurfaceDataset(self.test_sequences)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = True
        )

class SequenceModel(pl.LightningModule):
  def __init__(self, n_features, n_classes, n_hidden=400, n_layers=2):
    super().__init__()
    self.lstm = nn.LSTM(
        input_size = n_features,
        hidden_size = n_hidden,
        batch_first = True,
        num_layers = n_layers, 
        dropout = 0.3
    )
    self.liner1 = nn.Linear(n_hidden, n_classes)
    
  def forward(self, x):
    _, (hidden, _) = self.lstm(x)
    out = hidden[-1] 
    return out

def send(text):
  url = 'https://api.telegram.org/bot'+token+'/sendMessage?chat_id='+id+'&text='+text+''
  resp = requests.get(url)
  r = resp.json()

  return r

def edit(text):
  url = 'https://api.telegram.org/bot'+token+'/editMessage?chat_id='+id+'&text='+text+''
  resp = requests.get(url)
  r = resp.json()

  return r

def get_data():
    url = 'https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit={}'.format(SYMBOL, INTERVAL, LIMIT)
    res = requests.get(url)
    return_data = []
    for each in res.json():
        return_data.append(float(each[4]))
    return np.array(return_data)

def place_order(order_type):
    if(order_type == 'buy'):
        client.create_order(symbol=SYMBOL, side='buy', type='MARKET', quantity= QNTY)
        #send(f'Open position{order}')
    else:
        client.create_order(symbol=SYMBOL, side='sell', type='MARKET', quantity= QNTY)
        #send(f'Open position{order}')

class SurfacePredictor(pl.LightningModule):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, labels=None):
        output = self.model(x)
        if labels is not None:
            output = Variable(torch.randn(BATCH_SIZE, len(label_encoder.classes_)).float(), requires_grad=True).to(device)
        return output

    def test_step(self, batch, batch_idx):
        global predictions
        sequences = batch["sequence"]
        labels = batch["label"]
        outputs = self.forward(sequences, labels)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        print(predictions)

def condition_buy(price_current, price_buy, position):
    global number_of_trades, sell_state, buy, sell
    if price_buy + 1 + position > price_current >= price_buy + position:
        sell_state = position
        return 1
    elif (sell_state == position) and (price_current < price_buy + position):
        place_order('sell')
        send(f'Продаём. +{position}: {(100*price_current)/price_buy}\n Цена продажи: {price_current}')
        number_of_trades[position] += 1
        buy = False
        sell = True 
        sell_state = 0

        return 1    
    else: return 0

buy = False
sell = True
sell_state = 0
number_of_trades = np.zeros((32))

def main():
    global buy, sell, number_of_trades, sell_state, predictions
    model = SurfacePredictor(
        n_features=14,
        n_classes=2
    )
    model.load_state_dict(torch.load('model_weights_epochs_500.pth'))
    model.eval()    
    send(f' Что торгуем: {SYMBOL}\n С каким интервалом: {INTERVAL}\n Сколько : {QNTY}')

    info = client.get_account()
    df = pd.DataFrame(info["balances"])
    df["free"] = df["free"].astype(float).round(7)
    df = df[df["asset"] == 'USDT']
    balance = df["free"].to_numpy()
    send(f' Стартовый баланс:  {balance[0]}  USDT')

    while True:
        print(sell, buy)
        if sell == True: 
            test_sequences, features, label_encoder = dataset()
            data_module = SurfaceDataModule(0, test_sequences, BATCH_SIZE) 
            print('~~~~~ start testing:')
            pl.Trainer(accelerator='gpu',devices=1).test(model, data_module)
            print('~~~~~ finish')
            print(' predictions = ', predictions)
            if (predictions[0] == 1) and (buy == False):
                place_order('buy')
                price_buy = get_data()[-1]
                send(f' Покупаем\n Цена покупки: {price_buy}')
                buy = True
                sell = False
                oredr_sell = price_buy
                buy_time = time.time()
            if sell == False:
                time.sleep(1)
            else:
                time.sleep(60)

        if buy == True:
            print('Время продавать')
            while buy == True:
                price_current = get_data()[-1]
                if price_current >= price_buy + 31:
                    place_order('sell')
                    send(f'Продаём. + 9: {(100*price_current)/price_buy}\n Цена продажи: {price_current}')
                    number_of_trades[31] += 1
                    buy = False
                    sell = True
                elif ( time.time() - buy_time  > 60*30) and (price_current < oredr_sell):
                    place_order('sell')
                    send(f'Продаём GG: {(100*price_current)/price_buy}\n Цена продажи: {price_current}')
                    number_of_trades[0] += 1
                    buy = False
                    sell = True
                else:
                    condition_buy(price_current=price_current, price_buy=price_buy, position=30)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=29)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=28)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=27)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=26)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=25)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=24)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=23)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=22)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=21)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=20)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=19)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=18)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=17)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=16)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=15)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=14)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=13)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=12)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=11)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=10)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=9)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=8)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=7)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=6)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=5)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=4)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=3)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=2)
                    condition_buy(price_current=price_current, price_buy=price_buy, position=1)
                
                time.sleep(1)

            info = client.get_account()
            df = pd.DataFrame(info["balances"])
            df["free"] = df["free"].astype(float).round(7)
            df = df[df["asset"] == 'USDT']
            balance_current = df["free"].to_numpy()
            send(f' Баланс:  {balance_current[0]} USDT\n  {(balance_current[0]/balance[0]-1)*100} %\n Что по сделкам: {number_of_trades}')
        

schedule.every(1).seconds.do(main)

if __name__ == '__main__':
    main()
    