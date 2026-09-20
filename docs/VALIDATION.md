# Verification record

Verified locally on Windows with Python 3.12.14 on 20 September 2026.

## Automated checks

- **29 tests passed**, including real UCI-data integration; no tests skipped in the local run.
- Ruff lint and formatting checks passed; browser JavaScript syntax check passed.
- Training output reloaded from its saved artifact reproduces per-hour predictions and measured MAE.
- Tripling every final-test target leaves validation scores, calibrated radius and fitted model
  predictions on a fixed feature matrix unchanged. This tests the holdout boundary directly.
- Changing present/future target and weather values cannot change current or past feature vectors.
- A missing hourly timestamp invalidates the appropriate 1/24/168-hour lags rather than shifting
  the meaning of a row-based lag or introducing false zero demand.
- Invalid fields, missing features, nonfinite values, corrupt model files and conflicting labels
  are rejected. Prediction → feedback → MAE/coverage monitoring passes with a real SQLite file.
- Serving snapshots the matching model/report version; retraining cannot silently swap the
  report beneath a running old model.

Two upstream deprecation warnings were emitted by FastAPI/Starlette's test client (httpx and
AnyIO aliases). They do not affect the passing assertions. They are not suppressed.

## Browser checks

The running app was inspected in a Chromium browser:

- Desktop layout at 1440 × 1000 and mobile layout at 390 × 844; mobile document width remained
  within the viewport (375 px content width, 390 px viewport including scrollbar).
- Overview loads all 1,414 test records, four model metrics, seven slices and fifteen reliance values.
- Single-day selection for 1 November shows 24 hours; weekly selection shows its actual eligible
  sample count. Baseline labels match the 168-hour series.
- Forecast Lab performs a real local model call: the first historical example produces roughly
  45 rides, interval 0–128, and its actual model version. Modified inputs produce a hypothetical
  result and clear the prior output. No saved observation is claimed for edited scenarios.
- Observed monitoring shows 7/15 feature alerts; the synthetic shift shows 6/15. The text explains
  why moving winter values toward an autumn reference can reduce alerts.
- Methods marks only the best validation configuration (MAE 35.49), rather than every model
  sharing the same family name.
- A separate static-only server loads the report and explicitly disables new predictions.

Screenshots: [desktop](images/overview.png), [full page](images/overview-full.png),
[mobile](images/mobile.png). These show actual rendered saved metrics, not mock data.

## Reproduction and unverified scope

The real-data integration test trains into fresh temporary output directories. It reuses the
downloaded source and installed dependencies; it is **not** a fresh dependency installation.
GitHub Actions performs a clean Linux dependency install and full reproduction; see the
[workflow history](https://github.com/BelieveTZ/demandwatch-ml/actions/workflows/ci.yml) for results.

Docker is provided but was not executed on this host because Docker is not installed. No GPU,
load test, production deployment, data-arrival simulation or cross-city validation was performed.
