# ML Zoomcamp 2026 · Homework 5

The environment, model, and API come from the [official Homework 5 release](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/a395ca7ca995df92d8321daa017f3a1b4425a4a0/cohorts/2026/homework/05-deployment). `q3.py` checks the model checksum and predicts conversion for the Question 3 lead.

| Question | Answer | How it was checked |
| --- | --- | --- |
| 1. uv version | `0.10.11` | `uv --version` |
| 2. scikit-learn version | `1.7.2` | `uv.lock` and the installed version |
| 3. Loaded model | `0.533` | `q3.py` |
| 4. API | `0.770` | `POST /predict` returned `0.769799` and `conversion: true` |
| 5. Python base image | `python:3.11.15-slim-bookworm` | `FROM` in `Dockerfile` |
| 6. Container | `0.770` expected | Same model, API, and lead as Q4. Container execution pending. |

## Run the model and API

```bash
uv sync --locked
uv run --locked python q3.py
uv run --locked uvicorn predict:app --host 127.0.0.1 --port 9696
```

In a second terminal, from the homework folder, run the client and check the model checksum returned by the API.

```bash
uv run --locked python q6_test.py
curl --fail --silent http://127.0.0.1:9696/health
```

The local `/health` response returned `status: ok`. Both checksums matched `model_metadata.json`.

## Verify with Docker

Stop the local API to free port 9696. Start the container using the provided Dockerfile.

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 127.0.0.1:9696:9696 zoomcamp-model:2026-hw5
```

Run the same client and health commands from the second terminal. Compare the probability with Q4 and the model checksum with `model_metadata.json`.
