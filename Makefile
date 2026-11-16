.PHONY: install test test-all lint demo demo-all clean

install:
	pip install -e ".[data,dev]"

test:            ## fast tests, a few seconds
	python -m pytest -q -m "not slow"

test-all:        ## plus the end-to-end run on synthetic data (~1 min)
	python -m pytest -q

lint:
	python -m pyflakes src tests

demo:            ## synthetic run into results_demo/
	session-forecast --synthetic --fast --out results_demo

demo-all:        ## synthetic run with EVERY extra analysis (~5 min) into results_demo_all/
	session-forecast --synthetic --fast --all-extras --feature-set rich --out results_demo_all

clean:
	rm -rf results results_demo results_demo_all .pytest_cache build dist *.egg-info
