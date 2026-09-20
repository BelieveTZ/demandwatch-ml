# DemandWatch ML: method and evaluation notes

Research checked on 20 September 2026. These notes motivate the design; they do not claim experimental results. The generated evaluation report is the authority for the implemented split, fitted configuration, and measured scores.

## Dataset and provenance

The source is **Bike Sharing**, donated by Hadi Fanaee-T to the UCI Machine Learning Repository in 2013. It contains hourly and daily Capital Bikeshare rental counts from 2011–2012, with calendar and weather variables. Use `hour.csv` for this project. UCI identifies the dataset as **CC BY 4.0**. Preserve attribution with any redistributed data or derived sample. Dataset citation: Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894). [UCI source and license](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

UCI defines `cnt` as the total of `casual` and `registered`; therefore those two columns are target leakage and must never enter a demand predictor. `instant` is a record identifier, not a causal demand measurement. The landing page reports 17,389 instances; report the actual parsed file count, checksum, and timestamp range from ingestion rather than copying that metadata. [UCI variable definitions](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

## Prediction contract

Recommended scope: predict the next hourly demand count from calendar variables and observations already available before that hour. This is a project design choice, not a property guaranteed by the dataset.

- Calendar features for the target hour are available in advance.
- Lagged demand must come from an earlier timestamp. A rolling mean must exclude the target hour.
- Join a 24-hour or 168-hour lag by timestamp, not by row offset: missing hours would otherwise change the forecast horizon. Audit duplicate and absent timestamps. Do not silently interpret absent hours as zero demand.
- The dataset's target-hour weather represents observed weather. Using it is a weather-conditional estimate or an oracle-weather backtest, not a validated advance forecast. A forecast should use past weather, or separately sourced archived weather forecasts with their issue times.
- Using observed demand from earlier hours inside the test period is appropriate for a rolling one-hour-ahead evaluation when those counts would already have arrived. It does not demonstrate a 24-hour forecast made from one fixed origin.

The official scikit-learn bike-demand example constructs lagged predictors and demonstrates that shuffled evaluation is optimistic relative to chronological evaluation. Its published performance belongs to that example and must not be presented as this project's result. [Official forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).

## Recommended evaluation protocol

Use four non-overlapping chronological blocks after eligibility checks. One understandable calendar-based starting point is:

| Block | Suggested interval | Permitted use |
| --- | --- | --- |
| Training | 2011-01-01 through 2012-06-30 | Fit candidate models and learned preprocessing |
| Validation | 2012-07-01 through 2012-08-31 | Select configuration by the predeclared metric |
| Calibration | 2012-09-01 through 2012-10-31 | Estimate residual interval width for the frozen model |
| Test | 2012-11-01 through 2012-12-31 | Final point and interval evaluation |

Alternatively use fixed chronological proportions and record exact boundaries. These dates are recommendations, not assertions about the implementation. Choose boundaries before reviewing test scores. If refitting on training plus validation, do it **before** calibration, and do not refit that calibrated model afterward. Fit imputers, encoders, scaling, and baselines only on permitted fit data.

Expanding-window validation can reveal variability that one seasonal block hides. `TimeSeriesSplit` preserves order, grows the training window, and supports an excluded gap; equal sample spacing is required for folds with comparable durations. Missing hours make timestamp-based windows preferable to blindly counting rows. A gap should reflect target availability or forecast overlap, not serve as a substitute for correct features. [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Project reporting choices: compare a seasonal naive predictor and a simple learned baseline with the nonlinear model on identical eligible rows. Report MAE, RMSE, and sample counts; report improvement relative to a named baseline. Break down errors by hour and working-day status. Preserve poor slices and interval undercoverage instead of selecting only favorable results. Two years in one city cannot establish generalization to another city or today's system.

## Model choice

`HistGradientBoostingRegressor` supplies nonlinear tabular regression, native missing-value support, and squared-error, absolute-error, Poisson, and quantile objectives. Poisson loss uses a log link and accepts nonnegative targets. Automatic early stopping can enable an internal validation split above 10,000 samples. [Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

Recommended implementation: select a small, declared candidate set on chronological validation; set `early_stopping=False` and explicitly choose iteration counts, avoiding a hidden randomized time split. Pin package versions, estimator seeds, feature names, and settings in the run metadata. A compact CPU model is sufficient to demonstrate the full operating workflow; model size is not evidence of quality.

## Prediction intervals and their limits

For a frozen point predictor, compute absolute calibration residuals. With `n` residuals and miscoverage level `alpha`, use the order statistic at rank `ceil((n + 1) * (1 - alpha))`. If that rank exceeds `n`, reject the requested finite interval or return an unbounded interval; do not silently claim the desired guarantee. Apply the resulting symmetric radius to future point predictions, with a lower floor of zero for nonnegative counts. This is a split-conformal construction. [Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction*](https://arxiv.org/abs/2107.07511).

Ordinary marginal coverage guarantees rely on exchangeability; temporal dependence and changing demand violate that assumption. The interval in this project should be described as **calibrated on past residuals, with empirically evaluated coverage**, not as a guaranteed 90% interval for every future hour. Report the target level, test coverage, average width, calibration count, and slice coverage. Beyond-exchangeability methods address drift under additional constructions and assumptions; merely using recent residuals does not implement those methods. [Barber et al., *Conformal Prediction Beyond Exchangeability*](https://arxiv.org/abs/2202.13415).

## Explanation and monitoring

Permutation importance measures a fitted model's score degradation when a feature is shuffled; it can be computed on held-out data and repeated to estimate variability. Correlated features can mask one another, producing low individual importance even when the group matters. [scikit-learn permutation-importance guide](https://scikit-learn.org/stable/modules/permutation_importance.html).

Recommended presentation: show held-out MAE increase with repeat variability, name the evaluation block, and describe it as model reliance rather than causality. Shuffling temporal predictors can create unrealistic combinations; treat the chart as an approximate diagnostic. Do not use final-test explanations to iterate on features while continuing to call that test untouched.

Monitoring design choices: feature-distribution alerts indicate change, not proof of declining accuracy. Confirm performance degradation with delayed ground-truth labels; track errors and interval coverage separately. A simulated weather shift tests the alert mechanism, not a claim about real production traffic. Record reference and current windows, thresholds, and sample sizes so an alert can be investigated.
