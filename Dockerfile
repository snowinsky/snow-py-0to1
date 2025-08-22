FROM python:3.11

LABEL name="snow_py_0to1"
LABEL version="0.0.1"
LABEL description="call the ai api to do something"

WORKDIR /app

ADD . ./

# CMD ["python"]