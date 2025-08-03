install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	python -m black .

lint:
	python -m pylint --disable=R,C --exit-zero --recursive=y *

all: install format lint