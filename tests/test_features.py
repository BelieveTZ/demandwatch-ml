import numpy as np
import pandas as pd
import pytest

from demandwatch.data import FEATURES, load_source, make_features, split_blocks


@pytest.fixture
def hourly_source():
    index = pd.date_range("2011-01-01", periods=600, freq="h")
    return pd.DataFrame(
        {
            "cnt": np.arange(600, dtype=float),
            "workingday": 1,
            "holiday": 0,
            "temp": 0.5,
            "hum": 0.6,
            "windspeed": 0.2,
            "weathersit": 1,
            "casual": 99999,
            "registered": 99999,
        },
        index=index,
    )


def test_features_cannot_see_present_or_future(hourly_source):
    cutoff = hourly_source.index[400]
    before = make_features(hourly_source)
    changed = hourly_source.copy()
    changed.loc[cutoff:, ["cnt", "temp", "hum", "windspeed", "weathersit"]] = 9999
    after = make_features(changed)
    pd.testing.assert_frame_equal(before.loc[:cutoff, FEATURES], after.loc[:cutoff, FEATURES])
    assert not {"cnt", "casual", "registered", "temp", "instant"}.intersection(FEATURES)


def test_missing_hour_is_not_zero_or_previous_row(hourly_source):
    missing = hourly_source.index[300]
    frame = make_features(hourly_source.drop(missing))
    assert missing not in frame.index
    assert missing + pd.Timedelta(hours=1) not in frame.index
    assert missing + pd.Timedelta(hours=24) not in frame.index
    assert missing + pd.Timedelta(hours=168) not in frame.index
    target = hourly_source.index[350]
    assert frame.loc[target, "lag_24h"] == hourly_source.loc[target - pd.Timedelta(hours=24), "cnt"]
    expected_mean = hourly_source.loc[
        target - pd.Timedelta(hours=24) : target - pd.Timedelta(hours=1), "cnt"
    ].mean()
    assert frame.loc[target, "mean_24h"] == expected_mean


def test_chronological_blocks_are_disjoint():
    index = pd.date_range("2011-01-01", "2012-12-31 23:00", freq="h")
    blocks = split_blocks(pd.DataFrame({"target": 1}, index=index))
    for left, right in zip(list(blocks.values()), list(blocks.values())[1:], strict=False):
        assert left.index.max() < right.index.min()
    assert sum(len(block) for block in blocks.values()) == len(index)


@pytest.mark.parametrize("mutation", ["duplicate", "negative", "weather", "missing"])
def test_ingestion_rejects_bad_data(tmp_path, mutation):
    source = pd.DataFrame(
        {
            "dteday": ["2011-01-01"] * 2,
            "hr": [0, 1],
            "cnt": [1, 2],
            "workingday": [0, 0],
            "holiday": [0, 0],
            "temp": [0.5, 0.5],
            "hum": [0.6, 0.6],
            "windspeed": [0.2, 0.2],
            "weathersit": [1, 1],
        }
    )
    if mutation == "duplicate":
        source.loc[1, "hr"] = 0
    elif mutation == "negative":
        source.loc[0, "cnt"] = -1
    elif mutation == "weather":
        source.loc[0, "temp"] = 1.5
    else:
        source = source.drop(columns="cnt")
    path = tmp_path / "hour.csv"
    source.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_source(path)
