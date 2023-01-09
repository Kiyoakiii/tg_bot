
from aiogram.dispatcher import FSMContext
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters.state import State, StatesGroup
import inline_keyboard
import messages
import config
import json
from binance.client import Client
import binance 
import uuid
import os
import pika
from datetime import datetime
import uuid
import asyncio
from typing import MutableMapping
from aio_pika import Message, connect
import logging
from aio_pika.abc import (
    AbstractChannel, AbstractConnection, AbstractIncomingMessage, AbstractQueue,
)
from aiogram.contrib.fsm_storage.memory import MemoryStorage

import datetime

def getWeekDay():
    d=datetime.datetime.now()
    rst=d.isoweekday()
    return rst

time = getWeekDay()


class BinanceClient:
    connection: AbstractConnection
    channel: AbstractChannel
    callback_queue: AbstractQueue
    loop: asyncio.AbstractEventLoop

    def __init__(self):
        self.loop = asyncio.get_running_loop()

    async def call(self, symbol, id, api_key, api_secret, num_trades, stop):
        self.connection = await connect(os.environ['AMQP_URL'], loop=self.loop, timeout=60*60*24)
        async with self.connection:
            self.channel = await self.connection.channel()
            params = {
                "user_id": id,
                "symbol": symbol,
                "api_key": api_key,
                "api_secret": api_secret,
                "num_trades": num_trades,
                "stop": stop
            }
            jsonParams = json.dumps(params)
            await self.channel.default_exchange.publish(
                Message(
                    jsonParams.encode(),
                    content_type="application/json",
                ),
                routing_key="rpc_queue",
            )
        await self.connection.close()

async def num_of_trades(state:FSMContext ,id: int, message: str, api_key: str, api_secret: str, num_trades: int, stop: bool) -> None:
    binance_rpc = BinanceClient()
    await binance_rpc.call(message, id, api_key, api_secret, num_trades, stop)
    await state.finish()

async def stop(state:FSMContext ,id: int, message: str, api_key: str, api_secret: str, num_trades: int, stop: bool) -> None:
    binance_rpc = BinanceClient()
    await binance_rpc.call(message, id, api_key, api_secret, num_trades, stop)
    await state.finish()
class Gen(StatesGroup):
    wait_for_input_api_secret = State()
    wait_for_answer = State()
    wait_for_input_api_key = State()
    wait_for_input_num_of_trades = State()

logging.basicConfig(level=logging.INFO)

bot = Bot(token=config.BOT_API_TOKEN)
sum = 0
dp = Dispatcher(bot, storage=MemoryStorage())
api_key=''
api_secret=''

@dp.message_handler(commands=['start'])
async def show_info(message: types.Message):
    await message.reply(text=messages.reg(), reply_markup=inline_keyboard.REGISTRATION)

""" @dp.message_handler(commands=['stop'])
async def answer_on_input(message: types.Message, state: FSMContext):
    #await state.finish()
    await Gen.wait_for_answer.set()
    await stop(state, message.from_user.id, message = 'BTCUSDT', api_key= api_key, api_secret=api_secret, num_trades = message['text'], stop=True)
"""
@dp.message_handler(commands=['reg'])
async def show_info(message: types.Message):
    await Gen.wait_for_input_api_key.set()  
    await message.answer(messages.key_api())

@dp.message_handler(commands=['accountstatus'])
async def get_accountstatus(message: types.Message):
    client = Client(api_key, api_secret)
    status = client.get_account_status()
    await bot.send_message(message.chat.id, status)

@dp.message_handler(commands=['tradingstatus'])
async def get_tradingstatus(message: types.Message):
    client = Client(api_key, api_secret)
    status = client.get_account_api_trading_status()
    await bot.send_message(message.chat.id, status)

@dp.message_handler(commands=['trade'])
async def get_trade(message: types.Message):
    await message.answer(text=messages.trade(), reply_markup=inline_keyboard.PAIR)

@dp.callback_query_handler(text='BTCUSDT')
async def process_callback(message: types.Message, state: FSMContext):
    await bot.send_message(
        message.from_user.id,
        text=messages.num_trade(),
    )
    await state.finish()
    await Gen.wait_for_input_num_of_trades.set()  

@dp.message_handler(state=Gen.wait_for_input_num_of_trades)
async def answer_on_input_num_of_trades(message: types.Message, state: FSMContext):
    await state.finish()
    await Gen.wait_for_answer.set()
    await num_of_trades(state, message.from_user.id, message = 'BTCUSDT', api_key= api_key, api_secret=api_secret, num_trades = message['text'], stop=False)


#запрос на api_key
@dp.callback_query_handler(text='Привет')
async def process_callback_hi(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(
        callback_query.from_user.id,
        text=messages.menu()
    )

@dp.callback_query_handler(text='Заново')
async def process_callback_re(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await Gen.wait_for_input_api_key.set() 
    await bot.send_message(
        callback_query.from_user.id,
        text=messages.key_api()
) 

@dp.message_handler(state=Gen.wait_for_input_api_key)
async def answer_on_input_api_key(message: types.Message, state: FSMContext):
    global api_key
    await state.finish()
    await Gen.wait_for_input_api_secret.set()
    await message.answer('Хорошо, теперь введите свой api_secret')
    api_key = message['text']


@dp.message_handler(state=Gen.wait_for_input_api_secret)
async def answer_on_input_api_secret(message: types.Message, state: FSMContext):
    global api_secret
    api_secret = message['text']
    await connect_to_binance_acc(api_key, api_secret, message)
    await state.finish()

async def connect_to_binance_acc(api_key, api_secret, message):
    global sum
    try:
        Client(api_key, api_secret).get_account()
        await message.answer(f'Подключение прошло успешно')
    except binance.exceptions.BinanceAPIException or binance.exceptions.ConnectTimeout as e:
        if sum == 0:
            await message.answer(f'{e}\n' \
                f'Сначала введите api_key, после чего api_secret\n'\
                f'Разве это так сложно ???????')
            sum +=1
            await message.answer(text=messages.err(), reply_markup=inline_keyboard.TUPIT)
            
        elif sum == 1:
            await message.answer(f'Попытайтесь еще разок\n' \
                f'Сначала - api_key, потом - api_secret\n'\
                f'Помолимся чтобы в этот раз у вас все получилось')
            sum +=1
            await message.answer(text=messages.err(), reply_markup=inline_keyboard.TUPIT) 
        elif sum >= 2:
            await message.answer(text=messages.err(), reply_markup=inline_keyboard.TUPIT) 
            await message.answer(f'..... Даже с 3й попытки не получилось\n' \
                f'Lets go читать гайд\n'\
                f'https://www.binance.com/en/support/faq/how-to-create-api-360002502072')
    except UnicodeEncodeError:
        if sum == 0:
            await message.answer(f'{e}\n' \
                f'Сначала введите api_key, после чего api_secret\n'\
                f'Разве это так сложно ???????')
            sum +=1
            await message.answer(text=messages.err(), reply_markup=inline_keyboard.TUPIT)
            
        elif sum == 1:
            await message.answer(f'Попытайтесь еще разок\n' \
                f'Сначала - api_key, потом - api_secret\n'\
                f'Помолимся чтобы в этот раз у вас все получилось')
            sum +=1
            await message.answer(text=messages.err(), reply_markup=inline_keyboard.TUPIT) 
        elif sum >= 2:
            await message.answer(text=messages.err(), reply_markup=inline_keyboard.TUPIT) 
            await message.answer(f'..... Даже с 3й попытки не получилось\n' \
                f'Lets go читать гайд\n'\
                f'https://www.binance.com/en/support/faq/how-to-create-api-360002502072')

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
