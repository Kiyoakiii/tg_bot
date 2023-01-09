FROM python:3.8 as builder

WORKDIR /server/

ADD ./requirements.txt /server/requirements.txt
ADD ./model_weights_epochs_500.pth /server/model_weights_epochs_500.pth
RUN apt-get update 
RUN pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r ./requirements.txt

ADD . /server/
CMD ["python", "consumer.py"]