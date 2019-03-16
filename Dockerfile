FROM ubuntu:16.04
MAINTAINER ayman zah <aymanzah.github.io>

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas
RUN pip3 install matplotlib seaborn sklearn plotly

ADD ./boston_predict.py /opt/

ENTRYPOINT ["python3", "/opt/boston_predict.py"]
