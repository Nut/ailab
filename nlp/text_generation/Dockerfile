FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN apt-get update -y
RUN apt-get install -y build-essential

RUN pip install keras
RUN pip install pandas
RUN pip install -U spacy
RUN python -m spacy download en