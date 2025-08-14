

run:
	uv run src/arraysorting.py

pre-commit:
	uv run pre-commit run -a

pre-commit-manual:
	uv run pre-commit run -a --hook-stage manual


text_analysis_dev:
	uv run uvicorn src.text_analysis:app --reload --host 0.0.0.0 --port 8000


health:
	curl -X GET http://localhost:8000/health

test_text:
	curl -X 'POST' \
  'http://127.0.0.1:8000/count/words?dev=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{  "text": "test"}'
