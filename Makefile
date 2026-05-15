.PHONY: install ingest run

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -e .

ingest:
	. .venv/bin/activate && python -m app.main ingest

run:
	. .venv/bin/activate && python -m app.main run $(if $(HASH),--dedup-hash $(HASH),)
