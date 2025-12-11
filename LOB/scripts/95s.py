import pandas as pd
import os


def main():
    source_path = "data/LOB_scenarios_with_ci.csv"
    out_path = "data/lob_y_CI95_by_scenario.csv"

    df = pd.read_csv(source_path)

    # One row per scenario
    df_unique = df.drop_duplicates(subset=["Scenario"]).copy()

    out = pd.DataFrame()
    out["Scenario"] = df_unique["Scenario"]
    out["n_rep"] = df_unique["Replications"]

    def add_metric(prefix, mean_col, ci_col):
        out[f"{prefix}_mean"] = df_unique[mean_col]
        out[f"{prefix}_CI95_lower"] = df_unique[mean_col] - df_unique[ci_col]
        out[f"{prefix}_CI95_upper"] = df_unique[mean_col] + df_unique[ci_col]

    # Average spread
    add_metric("avg_spread", "avg_spread_Mean", "avg_spread_CI_95")

    # MEDIAN execution time (instead of mean)
    add_metric("median_exec_time", "median_exec_time_Mean", "median_exec_time_CI_95")

    # MM PnL per 1k trades
    add_metric(
        "mm_final_pnl_per_1k_trades",
        "mm_final_pnl_per_1k_trades_Mean",
        "mm_final_pnl_per_1k_trades_CI_95",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {out.shape}")


if __name__ == "__main__":
    main()
