FROM python:3.12.1

WORKDIR /work

COPY . /work

RUN pip install --upgrade pip

RUN pip install poetry &&\
    poetry config virtualenvs.create false

RUN poetry install