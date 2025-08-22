FROM python:3.11

WORKDIR /workspace

RUN apt-get update && apt-get install -y git build-essential

COPY . .
RUN pip install -r requirements.txt
RUN python -m pip cache purge
