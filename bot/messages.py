def reg() -> str:
    return f'Вас приветствует криптотелеграммбот\n' 

def err() -> str:
    return f'ГОЛОВА'

def key_api() -> str:
    return  f'Скиньте api_key, api_secret. Первым сообщением отправьте api_key(БЕЗ ПРОБЕЛОВ итд)' 

def api_key() -> str:
    return  f'Хорошо, введите api_key:' 

def api_secret() -> str:
    return  f'Хорошо, введите api_secret:' 

def menu() -> str:
    return  f'Загляни в меню и посмотри, что я могу сделать. Прежде всего /reg:' 

def num_trade() -> str:
    return f'Бот совершает фиксированное количество сделок. Отправьте сколько считаете нужным(>0)'

def trade() -> str:
    return f' Выберите торговую пару. Необходимо, чтобы было достаточно средств первого актива, даля торговли.'


