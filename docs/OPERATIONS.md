# Local operation and reproducibility

## Environment

Use Python 3.12 with `requirements.lock`. It records the installed environment,
including development tools. `pyproject.toml` declares supported dependency ranges;
the lockfile is the exact reproduction path. There are no inference credentials.

```bash
python -m venv .venv
# POSIX; on Windows use .venv\Scripts\python.exe
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m demandwatch.cli download
.venv/bin/python -m demandwatch.cli train
.venv/bin/python -m demandwatch.cli serve --port 8787
```

The CLI also supports `--source`, `--output`, `--report` on `train`, and `--artifacts`
on `serve`. Alternate sources must follow the UCI hourly schema and cover the fixed
calendar blocks; this is not a generic arbitrary-series training service.

## Outputs

- `data/raw/hour.csv` and `provenance.json`: source and download checksums (ignored).
- `artifacts/current/model.joblib`: fitted local model, interval radius and drift reference.
- `artifacts/current/manifest.json`: version/schema/configuration/hash contract.
- `artifacts/current/report.json` and `predictions.csv`: matching evaluation evidence.
- `demandwatch/web/data/`: published saved replay produced by training.
- `artifacts/current/predictions.sqlite`: local predictions and submitted labels (ignored).

`current` is a replaceable local output directory, not a full model registry. Use a
different output directory to preserve a run before retraining. Restart the server
after retraining: artifacts are loaded once at startup. A newly trained model's
version isolates its monitoring rows from older versions. The version hashes data,
configuration, sklearn and pipeline source; runtime timestamps and speed vary.

The server verifies the artifact checksum and sklearn version before loading. This
detects accidental corruption, **not malicious replacement**: joblib/pickle can execute
code, so load only artifacts trained locally from trusted source. Never download an
unknown model file and treat its adjacent checksum as authentication.

## API walkthrough

Interactive schema: [localhost:8787/docs](http://127.0.0.1:8787/docs).

```python
import json
import urllib.request

def request(path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        'http://127.0.0.1:8787' + path, data=payload,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req) as response:
        return json.load(response)

example = request('/data/report.json')['examples'][0]
prediction = request('/api/predict', {'features': example['features']})
print(prediction)
request('/api/feedback', {
    'prediction_id': prediction['prediction_id'],
    'actual': example['actual'],
})
print(request('/api/monitoring'))
```

This walkthrough explicitly replays a known historical observation. Real labels
would be sent only after the forecast hour ends. Inputs are prepared feature vectors,
not raw timestamps: callers must honor the same historical availability contract.
The editable UI makes hypothetical vectors and cannot validate their temporal
consistency. All feature keys are required; unknown keys, missing values, nonfinite
numbers, invalid calendar categories and out-of-range weather/counts are rejected.

| Endpoint | Behavior |
| --- | --- |
| `GET /api/health` | Saved-only or ready for local inference |
| `POST /api/predict` | Validated single-row prediction, interval, version and ID |
| `POST /api/feedback` | Add a ground-truth label; identical resubmission is idempotent |
| `GET /api/monitoring` | Drift and performance for latest 1,000 current-version predictions |
| `GET /data/report.json` | Active artifact report, or committed saved report without a model |
| `GET /data/predictions.csv` | Matching per-hour final-test evidence |

Feedback returns 404 for unknown IDs and 409 for conflicting labels. No model yields
503 for inference instead of fake predictions. Inference is serialized with one CPU
thread; the system is a local demonstration, not a throughput-optimized service.

## Docker

```bash
docker build -t demandwatch-ml .
docker run --rm -p 127.0.0.1:8787:8787 demandwatch-ml
```

The image build installs the lockfile, downloads checksum-verified data and trains
the model. It runs as a non-root user. The container binds internally to all
interfaces; the example exposes only host loopback. Logs disappear with `--rm`.
Docker requires internet access at build time. See VALIDATION.md for whether a
container build was actually exercised on the development host.

## Public replay

GitHub Pages hosts only `demandwatch/web/`, including attributed saved JSON/CSV.
The published UI disables new inference because Pages has no Python service. The
Verify workflow must succeed before the normal Pages workflow deploys that commit.
There is no public database, model server, user upload flow or paid hosting.

## Rebuilding evidence

```bash
python -m demandwatch.cli train
python scripts/export_evaluation.py
python -m pytest -q
```

Train replaces the published report; review metric, provenance and model-card changes
before committing. Do not tune against the frozen test and continue calling it untouched.
