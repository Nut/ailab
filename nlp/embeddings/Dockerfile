FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get update -y
RUN apt-get install -y build-essential
RUN apt-get install -y gcc

RUN pip install keras
RUN pip install gensim
RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn