FROM python:3.8

WORKDIR /bot/

ADD ./requirements.txt /bot/requirements.txt
RUN apt-get update
RUN pip install --no-cache-dir -r ./requirements.txt

ADD . /bot/
CMD ["python", "bot.py"]