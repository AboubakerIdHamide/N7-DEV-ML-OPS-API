"""Unit tests for feature selection, dataset validation, and cached loading."""

import pandas as pd
import pytest

from app.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    _select_features,
    _validate_dataset,
    load_credit_dataset,
)


def valid_dataset():
    """Build a two-row dataset containing both required target classes."""
    values = {
        column: [index + 1, index + 2]
        for index, column in enumerate(FEATURE_COLUMNS)
    }
    values[TARGET_COLUMN] = [0, 1]
    return pd.DataFrame(values)


def test_feature_selection_excludes_identifier_and_demographics():
    source = valid_dataset().drop(columns=[TARGET_COLUMN])
    source["ID"] = [1001, 1002]
    source["SEX"] = [1, 2]
    source["AGE"] = [25, 40]

    selected = _select_features(source)

    assert selected.columns.tolist() == FEATURE_COLUMNS
    assert "ID" not in selected.columns
    assert "SEX" not in selected.columns
    assert "AGE" not in selected.columns


def test_feature_selection_supports_uci_x_column_aliases():
    source = pd.DataFrame(
        {f"X{index}": [index, index + 1] for index in range(1, 24)}
    )
    source["ID"] = [1001, 1002]

    selected = _select_features(source)

    assert selected.columns.tolist() == FEATURE_COLUMNS
    assert selected.iloc[0]["LIMIT_BAL"] == 1
    assert selected.iloc[0]["PAY_0"] == 6
    assert selected.iloc[0]["BILL_AMT1"] == 12
    assert selected.iloc[0]["PAY_AMT6"] == 23


def test_dataset_validation_keeps_only_expected_columns_and_binary_target():
    dataset = valid_dataset()
    dataset["ID"] = [1001, 1002]

    cleaned = _validate_dataset(dataset)

    assert cleaned.columns.tolist() == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert cleaned[TARGET_COLUMN].tolist() == [0, 1]


def test_dataset_validation_rejects_missing_required_column():
    dataset = valid_dataset().drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing required columns"):
        _validate_dataset(dataset)


def test_dataset_validation_rejects_missing_values():
    dataset = valid_dataset()
    dataset.loc[0, FEATURE_COLUMNS[0]] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        _validate_dataset(dataset)


@pytest.mark.parametrize("labels", [[0, 0], [0, 2]])
def test_dataset_validation_requires_both_binary_labels(labels):
    dataset = valid_dataset()
    dataset[TARGET_COLUMN] = labels

    with pytest.raises(ValueError, match="binary target labels"):
        _validate_dataset(dataset)


def test_loader_reads_and_validates_cached_csv_without_fetching(tmp_path, monkeypatch):
    path = tmp_path / "selected.csv"
    valid_dataset().to_csv(path, index=False)

    def fail_if_fetch_is_called():
        """Fail the test if loading the cache tries to access UCI."""
        pytest.fail("cached dataset should be loaded without a network fetch")

    monkeypatch.setattr("app.data._fetch_from_uci", fail_if_fetch_is_called)

    loaded = load_credit_dataset(path)

    assert loaded.shape == (2, len(FEATURE_COLUMNS) + 1)
    assert loaded[TARGET_COLUMN].tolist() == [0, 1]
