#!/usr/bin/env python3
"""
Compute per-scenario means and 95% confidence intervals for all metrics,
and write an updated CSV with the *_Mean and *_CI_95 columns filled in.

Assumes the input CSV has columns in this order (at least):

    ... , sim_duration, ..., ask_level_5, Replications,
    sim_duration_Mean, sim_duration_CI_95, ..., ask_level_5_Mean, ask_level_5_CI_95

i.e. raw per-replication metrics from sim_duration through ask_level_5,
and then (possibly empty) summary columns with suffixes _Mean and _CI_95.

Usage:
    python compute_cis.py input.csv output.csv
"""

import argparse
import numpy as np
import pandas as pd


def compute_cis(
    df: pd.DataFrame,
    group_cols=("Scenario",),
    alpha: float = 0.05,
    z_value: float | None = None,
) -> pd.DataFrame:
    """
    For each scenario (group identified by group_cols), compute:

      - Replications: number of rows in the group
      - For each metric column between sim_duration and ask_level_5 (inclusive):
          * <metric>_Mean   = group mean
          * <metric>_CI_95  = half-width of 95% CI (mean ± half-width)

    CI is normal-based: mean ± z * (s / sqrt(n)), with z ≈ 1.96 for 95%.
    """

    # ------------------------------------------------------------------
    # 1. Identify metric columns: from sim_duration to ask_level_5 inclusive
    # ------------------------------------------------------------------
    try:
        start_idx = df.columns.get_loc("sim_duration")
        end_idx = df.columns.get_loc("ask_level_5")
    except KeyError as e:
        raise KeyError(
            "Expected columns 'sim_duration' and 'ask_level_5' "
            "to exist in the input CSV."
        ) from e

    metric_cols = df.columns[start_idx : end_idx + 1]

    # ------------------------------------------------------------------
    # 2. Group by scenarios (or scenario + parameters, if desired)
    # ------------------------------------------------------------------
    group_cols = list(group_cols)
    g = df.groupby(group_cols, dropna=False)

    # Number of replications per group
    # (use any column; Replication is available)
    n = g["Replication"].transform("count")
    df["Replications"] = n

    # ------------------------------------------------------------------
    # 3. CI calculation
    # ------------------------------------------------------------------
    if z_value is None:
        # Normal approximation for 95% CI
        z_value = 1.96 if abs(alpha - 0.05) < 1e-9 else np.nan

    # Avoid division by zero / undefined SE when n <= 1
    n_for_se = n.where(n > 1, np.nan)

    for col in metric_cols:
        mean_col = f"{col}_Mean"
        ci_col = f"{col}_CI_95"

        # Group-wise mean
        mean_vals = g[col].transform("mean")

        # Sample standard deviation with ddof=1
        std_vals = g[col].transform("std", ddof=1)

        # Standard error and half-width of CI
        se_vals = std_vals / np.sqrt(n_for_se)
        ci_half_width = z_value * se_vals

        df[mean_col] = mean_vals
        df[ci_col] = ci_half_width

    return df


def main() -> None:
    input_csv = "data/simulation_results_full.csv"
    output_csv = "data/LOB_scenarios_with_ci.csv"

    df = pd.read_csv(input_csv)
    df_out = compute_cis(df)
    df_out.to_csv(output_csv, index=False)



if __name__ == "__main__":
    main()
