test:
	pytest tests/ -v

backtest:
	python -m backtest.engine --config config/strategies/trend_following_v1.yaml

parity-check:
	pytest tests/integration/test_full_chain_golden_path.py -v

lint:
	black src/ tests/ research/ --check
	flake8 src/ tests/ research/
