FROM ubuntu:20.04
MAINTAINER Kudrina Alisa
RUN apt-get update -y
COPY . /opt/gsom_predictor
WORKDIR /opt/gsom_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD pytho3 app.py

