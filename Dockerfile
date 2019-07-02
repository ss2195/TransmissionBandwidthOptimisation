FROM ubuntu:16.04

MAINTAINER Shubham "shubhamshourya1995@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python3-pip

#RUN apt-get update && apt-get install -y apt-transport-https

#RUN apt-get install -y software-properties-common

#RUN add-apt-repository ppa:jonathonf/python-3.6

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

#RUN pip3 install --upgrade pip3

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
