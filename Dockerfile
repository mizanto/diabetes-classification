FROM python:3.11.5-slim

WORKDIR /app

RUN pip --no-cache-dir install pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --deploy --system

COPY ./src /app/src
COPY ./data /app/data
COPY ./models /app/models

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "src.service:app"]