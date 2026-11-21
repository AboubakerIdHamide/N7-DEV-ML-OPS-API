"""Tests for the local, DVC-compatible dataset preparation command."""

import pandas as pd

from app.data import FEATURE_COLUMNS, TARGET_COLUMN
from app.prepare_data import main


def test_prepare_data_writes_selected_dataset_without_network(
    tmp_path, monkeypatch, capsys
):
    """Run the DVC entry point using a local sample and a temporary output path."""
    output_path = tmp_path / "raw" / "selected.csv"
    sample = pd.DataFrame(
        {
            **{
                column: [index + 1, index + 2]
                for index, column in enumerate(FEATURE_COLUMNS)
            },
            TARGET_COLUMN: [0, 1],
        }
    )
    monkeypatch.setenv("CREDIT_DATA_PATH", str(output_path))
    monkeypatch.setattr("app.data._fetch_from_uci", lambda: sample)

    main()

    prepared = pd.read_csv(output_path)
    assert prepared.columns.tolist() == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert prepared[TARGET_COLUMN].tolist() == [0, 1]
    assert f"at {output_path}." in capsys.readouterr().out
