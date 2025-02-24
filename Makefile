.PHONY: test

test:
	pytest --cov=dask_mwu --cov-report=html -n auto
