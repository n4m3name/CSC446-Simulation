import os
from itertools import combinations

import pandas as pd


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    means_raw = pd.read_csv("data/simulation_results_means.csv", header=None)
    ci = pd.read_csv("data/lob_y_CI95_by_scenario.csv")

    # ------------------------------------------------------------------
    # 2. Extract factor levels
    # ------------------------------------------------------------------
    factors = means_raw[[0, 3, 4, 5]].copy()
    factors.columns = [
        "Scenario",
        "order_size_mean",
        "limit_price_decay_rate",
        "mm_base_spread",
    ]

    # ------------------------------------------------------------------
    # 3. Merge
    # ------------------------------------------------------------------
    df = factors.merge(ci, on="Scenario", how="inner")

    if df.shape[0] != 125:
        print(f"Warning: expected 125 scenarios, got {df.shape[0]} rows after merge.")

    # ------------------------------------------------------------------
    # 4. Output directories
    # ------------------------------------------------------------------
    data_dir = "data/analysis"
    figs_dir = "figs/analysis"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Save merged dataset
    scenario_out = os.path.join(data_dir, "analysis_scenario_level.csv")
    df.to_csv(scenario_out, index=False)

    # ------------------------------------------------------------------
    # 5. Metrics (use mean execution time instead of price volatility)
    # ------------------------------------------------------------------
    metrics = {
        "avg_spread": {
            "mean_col": "avg_spread_mean",
            "lower_col": "avg_spread_CI95_lower",
            "upper_col": "avg_spread_CI95_upper",
        },
        "mean_exec_time": {
            "mean_col": "mean_exec_time_mean",
            "lower_col": "mean_exec_time_CI95_lower",
            "upper_col": "mean_exec_time_CI95_upper",
        },
        "mm_pnl_per_1k_trades": {
            "mean_col": "mm_final_pnl_per_1k_trades_mean",
            "lower_col": "mm_final_pnl_per_1k_trades_CI95_lower",
            "upper_col": "mm_final_pnl_per_1k_trades_CI95_upper",
        },
    }

    factors_list = ["order_size_mean", "limit_price_decay_rate", "mm_base_spread"]

    # ------------------------------------------------------------------
    # 6. Main effects
    # ------------------------------------------------------------------
    for metric_name, spec in metrics.items():
        mean_col = spec["mean_col"]

        for factor in factors_list:
            main_effect = (
                df.groupby(factor)[mean_col]
                .agg(["mean"])
                .reset_index()
                .rename(columns={"mean": f"{metric_name}_mean"})
            )

            out_path = os.path.join(
                data_dir, f"main_effect_{metric_name}_by_{factor}.csv"
            )
            main_effect.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # 7. Two-factor interactions
    # ------------------------------------------------------------------
    for metric_name, spec in metrics.items():
        mean_col = spec["mean_col"]

        for f1, f2 in combinations(factors_list, 2):
            interaction = df.pivot_table(
                values=mean_col,
                index=f1,
                columns=f2,
                aggfunc="mean",
            )

            out_path = os.path.join(
                data_dir, f"interaction_{metric_name}_by_{f1}_x_{f2}.csv"
            )
            interaction.to_csv(out_path)

    # ------------------------------------------------------------------
    # 8. Scenario rankings
    # ------------------------------------------------------------------
    rank_cols = [
        "Scenario",
        "order_size_mean",
        "limit_price_decay_rate",
        "mm_base_spread",
    ]

    for metric_name, spec in metrics.items():
        mean_col = spec["mean_col"]

        ranked = df.sort_values(mean_col, ascending=True)

        bottom5 = ranked.head(5)[rank_cols + [mean_col]]
        top5 = ranked.tail(5)[rank_cols + [mean_col]].iloc[::-1]

        bottom_out = os.path.join(data_dir, f"{metric_name}_bottom5_scenarios.csv")
        top_out = os.path.join(data_dir, f"{metric_name}_top5_scenarios.csv")

        bottom5.to_csv(bottom_out, index=False)
        top5.to_csv(top_out, index=False)

    # ------------------------------------------------------------------
    # 9. CI width summaries
    # ------------------------------------------------------------------
    ci_summary_rows = []
    for metric_name, spec in metrics.items():
        lower_col = spec["lower_col"]
        upper_col = spec["upper_col"]

        width = df[upper_col] - df[lower_col]

        ci_summary_rows.append(
            {
                "metric": metric_name,
                "mean_CI_width": width.mean(),
                "max_CI_width": width.max(),
                "min_CI_width": width.min(),
            }
        )

        width_df = df[["Scenario"]].copy()
        width_df["CI_width"] = width
        out_path = os.path.join(
            data_dir, f"{metric_name}_CI_widths_by_scenario.csv"
        )
        width_df.to_csv(out_path, index=False)

    ci_summary = pd.DataFrame(ci_summary_rows)
    ci_summary_out = os.path.join(data_dir, "CI_width_summary_overall.csv")
    ci_summary.to_csv(ci_summary_out, index=False)


if __name__ == "__main__":
    main()