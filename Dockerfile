FROM python:3.11-slim

WORKDIR /app
COPY . /app/

RUN pip3 install -r requirements.txt

# Running pylint on all python files
RUN pylint --fail-under 5 ./src

# Runing unit tests
RUN pytest src/tests/

CMD ["fastapi","run","src/main.py"]