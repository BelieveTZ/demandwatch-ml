# Evaluation record

Run `cc2284354e35`. Recorded at 2026-09-20T01:54:32.772807+00:00. Python 3.12, CPU only.

The checked-in [JSON](../demandwatch/web/data/report.json) and [per-hour CSV](../demandwatch/web/data/predictions.csv) are the measured evidence.

## Source and eligibility

Parsed 17,379 source rows with 165 absent hourly timestamps. 16,886 eligible rows remain after strictly historical feature requirements. The first week and incomplete histories are excluded.

CSV SHA-256: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.

## Frozen protocol

| Block | Rows | Start | End |
| --- | ---: | --- | --- |
| train | 12,580 | 2011-01-08 00:00:00 | 2012-06-30 23:00:00 |
| validation | 1,488 | 2012-07-01 00:00:00 | 2012-08-31 23:00:00 |
| calibration | 1,404 | 2012-09-01 00:00:00 | 2012-10-31 23:00:00 |
| test | 1,414 | 2012-11-01 00:00:00 | 2012-12-31 23:00:00 |

Four candidates (one Ridge, three HGB configurations) are ranked by validation MAE. The winner is refit on train + validation, frozen, then calibrated. The final test does not select parameters or interval width.

| Candidate | Validation MAE | Configuration |
| --- | ---: | --- |
| Ridge | 48.234 | `{"alpha": 10}` |
| Histogram gradient boosting | 36.257 | `{"l2_regularization": 1.0, "learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 15}` |
| Histogram gradient boosting | 35.489 | `{"l2_regularization": 1.0, "learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 31}` |
| Histogram gradient boosting | 37.188 | `{"l2_regularization": 5.0, "learning_rate": 0.05, "max_iter": 350, "max_leaf_nodes": 31}` |

## Final test

| Model | MAE | RMSE | Hours |
| --- | ---: | ---: | ---: |
| Seasonal 24h | 64.267 | 107.357 | 1,414 |
| Seasonal 168h | 72.677 | 118.636 | 1,414 |
| Ridge | 42.510 | 63.811 | 1,414 |
| Histogram gradient boosting | 24.368 | 37.984 | 1,414 |

Selected: **Histogram gradient boosting**. MAE reduction versus Seasonal 168h: **66.47%** on identical eligible rows.

90% target interval: empirical test coverage **94.98%**, mean total width **147.09** rides. Residual radius **82.75**, fitted using 1,404 calibration hours. The nonnegative floor means mean total width can be less than twice the radius.

## Error slices

| Slice | Hours | MAE | RMSE | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Working day | 912 | 24.21 | 38.19 | 95.07% |
| Non-working day | 502 | 24.65 | 37.60 | 94.82% |
| Morning peak (07–09) | 177 | 36.40 | 52.83 | 87.57% |
| Evening peak (16–19) | 240 | 40.11 | 53.85 | 87.92% |
| Overnight (00–05) | 341 | 6.89 | 15.85 | 99.41% |
| November | 677 | 26.46 | 41.44 | 93.94% |
| December | 737 | 22.45 | 34.51 | 95.93% |

Peak-hour coverage falls below the nominal target even though aggregate coverage is higher. A single absolute-error radius overcovers quiet hours and undercovers busy ones. This run retains that failure rather than retuning on the test period.

## Timing and monitoring

Warm single-row prediction: p50 **1.426 ms**, p95 **1.645 ms**, n=100. One CPU thread; includes DataFrame feature selection and estimator prediction; excludes network, HTTP parsing, SQLite and cold start.

Platform: Windows-11-10.0.26200-SP0; processor: AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD.

PSI uses calibration-period fixed quantile bins, explicit out-of-support bins and 0.5-count smoothing. 0.2 is a review heuristic. Calendar/time trend shifts are expected. The synthetic monitor test adds 0.25 to prior temperature (clipped to [0,1]) and multiplies demand lags/means by 1.6. It is not an accuracy experiment.

## Limits

- Rolling one-hour-ahead backtest, assuming previous-hour counts arrive immediately.
- Previous-hour weather only; no observed target-hour weather or target components.
- Temporal dependence: 90% is a calibration target, not a coverage guarantee.
- Missing source hours and incomplete lag histories are excluded, not zero-filled.
- Two historical years in one city; no current-system or cross-city validation.
- PSI thresholds are heuristics. Seasonal calendar drift can be expected.
- Synthetic stress shifts exercise monitors; they are not live production incidents.
- Latency is local, warm, single-threaded model time, not an API service-level claim.

Validation permutation importance uses the train-only winner; correlated lag features make this approximate model reliance, not causal explanation. No city-transfer, probabilistic calibration under drift, or multi-day forecast claim.
