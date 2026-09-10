test:
	pytest tests/ -v

backtest:
	python -m backtest.engine --config config/strategies/trend_following_v1.yaml

parity-check:
	pytest tests/integration/test_full_chain_golden_path.py -v

lint:
	black src/ tests/ research/ --check
	flake8 src/ tests/ research/

# --- Lớp điều hướng cho AI agent ---
codemap:
	python scripts/gen_codemap.py

check-docs:
	python scripts/check_docs.py

agent-sync: codemap check-docs
