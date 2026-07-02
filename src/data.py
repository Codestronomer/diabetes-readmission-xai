import os
import hashlib
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo


def fetch_and_verify_data(save_dir="../data/raw", filename="diabetes_130_us_raw.csv"):
    """
    Fetches the UCI Diabetes 130-US hospitals dataset, verifies its integrity,
    and saves it to the specified raw data directory.
    """
    print("Fetching Diabetes 130-US hospitals dataset from UCI (ID: 296)...")

    try:
        # Fetch dataset
        diabetes = fetch_ucirepo(id=296)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    # Extract features and targets, then combine
    ids = diabetes.data.ids
    X = diabetes.data.features
    y = diabetes.data.targets
    df = pd.concat([ids, X, y], axis=1)

    # Blueprint Verification Gate 1: Check dimensions
    expected_rows = 101766
    expected_cols = 50  # 49 features + 1 target

    print("\n--- Verification Gate 1 ---")
    print(f"Actual Shape:   {df.shape}")
    print(f"Expected Shape: ({expected_rows}, {expected_cols})")

    if df.shape[0] == expected_rows and df.shape[1] == expected_cols:
        print("Status: PASSED (Dimensions match perfectly).")
    else:
        print(
            "Status: FAILED (Dimensions do not match!). Please investigate before proceeding."
        )

    # Save to raw data directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / filename

    df.to_csv(file_path, index=False)

    print(f"Saved to: {file_path.resolve()}")
    print(f"Exists: {file_path.exists()}")

    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"SHA256: {file_hash}")
    print(f"SHA256 Checksum: {file_hash}")
    print("---------------------------\n")

    return df


if __name__ == "__main__":
    fetch_and_verify_data()
