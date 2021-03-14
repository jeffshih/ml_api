From python:3.8.6-slim-buster
MAINTAINER jeffshih <jeffhfs1224@gmail.com>
RUN apt-get update && apt-get install -y\
        libpq-dev\
        build-essential
EXPOSE 80
WORKDIR /MLSERVER
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
