from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

CLIENT_REGISTERED = InlineKeyboardButton('Привет', callback_data='Привет')
REGISTRATION = InlineKeyboardMarkup().add(CLIENT_REGISTERED)

api_key = InlineKeyboardButton('api_key', callback_data='api_key')
api_secret = InlineKeyboardButton('api_secret', callback_data='api_key')
KEY = InlineKeyboardMarkup().add(api_key, api_secret)

CLIENT_TUPIT = InlineKeyboardButton('Заново', callback_data='Заново')
TUPIT = InlineKeyboardMarkup().add(CLIENT_TUPIT)

USDTBTC = InlineKeyboardButton('USDT/BTC', callback_data='BTCUSDT')
PAIR = InlineKeyboardMarkup().add(USDTBTC)