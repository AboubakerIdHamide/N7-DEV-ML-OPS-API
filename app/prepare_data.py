"""Prepare the selected public dataset as a reproducible DVC stage."""

from __future__ import annotations

import os
from pathlib import Path

from app.data import DEFAULT_DATA_PATH, load_credit_dataset


def main() -> None:
    """Prepare the cached/remote dataset and report the resulting table dimensions."""
    data_path = Path(os.getenv("CREDIT_DATA_PATH", DEFAULT_DATA_PATH))
    dataset = load_credit_dataset(data_path)
    print(
        f"Prepared {len(dataset)} rows and {len(dataset.columns)} columns "
        f"at {data_path}."
    )


if __name__ == "__main__":
    main()
