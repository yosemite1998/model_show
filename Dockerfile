# syntax=docker/dockerfile:1
FROM python:3.8.12
#FROM tensorflow/tensorflow:latest
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# RUN apk update
# RUN apk add --no-cache gcc g++ musl-dev linux-headers git
COPY requirements.txt requirements.txt
RUN apt update && apt install -y libgl1-mesa-glx
RUN pip install --upgrade pip && pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN mkdir /temp
EXPOSE 5000
COPY . .
CMD ["flask", "run"]