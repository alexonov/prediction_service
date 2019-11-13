FROM docker.io/python:3.7

COPY src/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/ /app
WORKDIR /app

CMD ["gunicorn", "-b", "0.0.0.0:9000", "app:api"]