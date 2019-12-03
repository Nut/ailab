FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

#USER root
RUN apt-get update -y
RUN apt-get install -y build-essential

#USER $NB_USER
RUN pip install keras
RUN pip install -U spacy
RUN python -m spacy download en
