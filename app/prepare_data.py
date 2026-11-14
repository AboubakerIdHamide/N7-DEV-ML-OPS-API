"""Prepare the selected public dataset as a reproducible DVC stage."""

from __future__ import annotations

from app.data import DEFAULT_DATA_PATH, load_credit_dataset


def main() -> None:
    dataset = load_credit_dataset()
    print(
        f"Prepared {len(dataset)} rows and {len(dataset.columns)} columns "
        f"at {DEFAULT_DATA_PATH}."
    )


if __name__ == "__main__":
    main()
