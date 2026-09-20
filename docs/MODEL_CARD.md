# Model card

## Intended use

Research and portfolio demonstration of rolling one-hour-ahead total rental demand
prediction for the historical Capital Bikeshare system. Supports tabular regression,
temporal evaluation, uncertainty diagnostics and local serving demonstrations.

Not a station allocation optimizer, per-customer model, real-time deployed forecast,
or a validated forecast for current or other-city demand.

## Data and availability

17,379 observed hourly rows from 2011–2012; 165 missing timestamps. Missing hours
remain missing. No personal rider records are used. Eligibility requires a target,
1/24/168-hour count lags, historical rolling means and previous-hour weather.
16,886 rows are eligible. Rolling means require at least 12/24 or 84/168 observed
hours respectively; missing observations are ignored within those windows.

At origin t, the contract assumes counts/weather through t−1 are finalized and
calendar attributes at t are known. Counts, weather or holidays arriving late in a
real system would require a different availability policy and a new backtest.
All timestamps are source-local naive hourly labels; duplicate labels are rejected.
No timezone/DST reconciliation beyond the supplied UCI record is implemented.

## Estimator and selection

One standardized/one-hot Ridge candidate (alpha 10) and three Poisson-loss histogram
gradient boosting configurations are fixed in `training.py`. HGB automatic early
stopping is disabled to avoid a random internal holdout. Random seed 42, one CPU
thread. Validation MAE selects the winner. Final estimator refits on train +
validation before calibration; the test period does not influence that choice.

Read [EVALUATION.md](EVALUATION.md) for exact dates, parameters and measured results.
Artifact manifests record feature order, sklearn version, source hash, pipeline
source hash, configuration and model-file checksum.

## Interval construction

The radius is the finite-sample corrected order statistic of absolute calibration
residuals. Prediction is clipped at zero; lower bound is also clipped at zero. Target
coverage is 90%. Temporal dependence and drift violate ordinary exchangeability:
report empirical coverage rather than a distribution-free temporal guarantee.
Symmetric constant-width intervals miss heteroscedastic peak-hour behavior.

## Monitoring

Calibration-period quantile bins with explicit extreme bins and 0.5-count smoothing
define PSI reference distributions. A score of at least 0.2 prompts review; under
100 current observations yields `insufficient_data`. Correlated features, seasonality
and monotonically advancing time can all cause expected alarms. PSI is neither a
p-value nor a measure of predictive accuracy. Labels, when submitted, yield MAE,
RMSE and interval coverage for the latest 1,000 local predictions of the current
model version. Repeated what-if predictions are not representative traffic.

## Risks and next research step

The held-out block contains only two winter months. The validation/calibration
seasons differ; temporal/city transfer is unmeasured. Aggregate intervals hide peak
undercoverage. A future experiment should preregister rolling origins and compare
quantile-based conditional intervals on a newly reserved period or dataset, rather
than improve this frozen test by repeatedly inspecting it.
