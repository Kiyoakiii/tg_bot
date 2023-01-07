
from aiogram.dispatcher import FSMContext
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.redis import RedisStorage2
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

from aio_pika import Message, connect
import logging


import datetime

def getWeekDay():
    d=datetime.datetime.now()
    rst=d.isoweekday()
    return rst

time = getWeekDay()


class BinanceClient(object):

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.connection = connect(
            os.environ['AMQP_URL'], loop=self.loop, channel_number=1,
        )
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    async def call(self, symbol, id, api_key, api_secret):
        #message =  api_key + '|' + api_secret + '|s' + symbol + '+' + id
        self.response = None
        self.corr_id = str(uuid.uuid4())
        params = {
            "user_id": id,
            "symbol": symbol,
            "api_key": api_key,
            "api_secret": api_secret,
            "text": ''
        }
        jsonParams = json.dumps(params)
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue_tg_4',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=jsonParams.encode())
        self.connection.process_data_events(time_limit=None)
        return (self.response)

async def get_answer_and_reply(id: int, message: str, api_key: str, api_secret: str, num_trades: int) -> None:
    binance_rpc = BinanceClient()
    response = await binance_rpc.call(message, id, api_key, api_secret)
    answer = json.loads(response.decode("UTF-8"))
    await bot.send_message(answer['user_id'], answer['message'])

async def qwe(text: str, id: int, api_key: str, api_secret: str) -> int:
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    connection = pika.BlockingConnection(pika.ConnectionParameters(
               'localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=f'rpc_queue_tg_4', durable=True) #Создаем очередь
    message =  api_key + '|' + api_secret + '|s' + text + '+' + id
    channel.basic_publish(exchange='',
                      body=message,
                      routing_key='rpc_queue_tg_4',
                      properties=pika.BasicProperties(
                        delivery_mode = 2, # make message persistent
                      ))

    return await future

class Gen(StatesGroup):
    wait_for_input_api_secret = State()
    wait_for_answer = State()
    wait_for_input_api_key = State()
    wait_for_input_num_of_trades = State()

logging.basicConfig(level=logging.INFO)

bot = Bot(token=config.BOT_API_TOKEN)
#dp = Dispatcher(bot)
sum = 0
dp = Dispatcher(bot, storage=RedisStorage2(host=os.environ['REDIS_HOST'], db=1, port=os.environ['REDIS_PORT'], password=os.environ['REDIS_PASSWORD']))

api_key=''
api_secret=''

@dp.message_handler(commands=['start'])
async def show_info(message: types.Message):
    await message.reply(text=messages.reg(), reply_markup=inline_keyboard.REGISTRATION)

@dp.message_handler(commands=['reg'])
async def show_info(message: types.Message):
    await Gen.wait_for_input_api_key.set()  
    await message.answer(messages.key_api())

@dp.message_handler(commands=['accountstatus'])
async def all_orders(message: types.Message):
    client = Client(api_key, api_secret)
    status = client.get_account_status()
    await bot.send_message(message.chat.id, status)

@dp.message_handler(commands=['tradingstatus'])
async def all_orders(message: types.Message):
    client = Client(api_key, api_secret)
    status = client.get_account_api_trading_status()
    await bot.send_message(message.chat.id, status)

@dp.message_handler(commands=['trade'])
async def all_orders(message: types.Message):
    await message.answer(text=messages.trade(), reply_markup=inline_keyboard.PAIR)

@dp.callback_query_handler(text='BTCUSDT')
async def process_callback(message: types.Message):
    await bot.send_message(
        message.from_user.id,
        text=messages.num_trade(),
    )
    await Gen.wait_for_input_num_of_trades.set()  

@dp.message_handler(state=Gen.wait_for_input_num_of_trades)
async def answer_on_input(message: types.Message, state: FSMContext):
    await get_answer_and_reply(message.from_user.id, message = 'BTCUSDT', api_key= api_key, api_secret=api_secret, num_trades = message)
    await state.finish()

#запрос на api_key
@dp.callback_query_handler(text='Привет')
async def process_callback_weather(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(
        callback_query.from_user.id,
        text=messages.menu()
    )

@dp.callback_query_handler(text='Заново')
async def process_callback_weather(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await Gen.wait_for_input_api_key.set() 
    await bot.send_message(
        callback_query.from_user.id,
        text=messages.key_api()
) 

@dp.message_handler(state=Gen.wait_for_input_api_key)
async def answer_on_input(message: types.Message, state: FSMContext):
    print('csdcsd')
    global api_key
    await Gen.wait_for_input_api_secret.set()
    await message.answer('Хорошо, теперь введите свой api_secret')
    api_key = message['text']


@dp.message_handler(state=Gen.wait_for_input_api_secret)
async def answer_on_input(message: types.Message, state: FSMContext):
    global api_secret
    api_secret = message['text']
    await connect_to_binance_acc(api_key, api_secret, message)
    await state.finish()

@dp.message_handler(content_types=['get_all_orders'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Что как и почему")
    btn2 = types.KeyboardButton("❓ Задать вопрос")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, text="Привет, {0.first_name}! Я тестовый бот для твоей статьи для habr.com".format(message.from_user), reply_markup=markup)

async def connect_to_binance_acc(api_key, api_secret, message):
    global sum
    try:
        Client(api_key, api_secret).get_account()
        await message.answer(f'Подключение прошло успешно')
    except binance.exceptions.BinanceAPIException as e:
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
