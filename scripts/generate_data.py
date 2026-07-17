#!/usr/bin/env python3
"""Generate simulated longitudinal data for the HBM + ML pipeline."""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulate import create_simulated_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate simulated train/test data")
    parser.add_argument("--n-train", type=int, default=800, help="Number of training individuals")
    parser.add_argument("--n-test", type=int, default=200, help="Number of test individuals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)

    n_total = args.n_train + args.n_test

    print(f"Generating {n_total} individuals ({args.n_train} train, {args.n_test} test)...")
    df = create_simulated_dataset(n_individuals=n_total)

    patient_ids = df["id"].unique()
    np.random.shuffle(patient_ids)
    train_ids = set(patient_ids[: args.n_train])
    test_ids = set(patient_ids[args.n_train :])

    train_df = df[df["id"].isin(train_ids)].copy()
    test_df = df[df["id"].isin(test_ids)].copy()

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} rows, {train_df['id'].nunique()} patients -> {train_path}")
    print(f"Test:  {len(test_df)} rows, {test_df['id'].nunique()} patients -> {test_path}")


if __name__ == "__main__":
    main()
