FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y && \
    apt-get update && \
    pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "app.py" ]