

run:
	uv run src/arraysorting.py

pre-commit:
	uv run pre-commit run -a

pre-commit-manual:
	uv run pre-commit run -a --hook-stage manual
