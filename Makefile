default:

install:
	pip install -r requirements.txt
	pip install --use-pep517 -e .

test:
	pytest
	make -C docs doctest

format:
	black mushi
	docformatter --in-place mushi/*.py

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 . --count --max-complexity=30 --max-line-length=127 --statistics

docs:
	make -C docs html

.PHONY: install test format lint deploy docs
