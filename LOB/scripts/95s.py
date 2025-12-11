import pandas as pd
import os

def main():
    # Path to the "wide" summary file whose header you pasted
    source_path = "data/LOB_scenarios_with_ci.csv"  # <-- change if needed
    out_path = "data/lob_y_CI95_by_scenario.csv"

    df = pd.read_csv(source_path)

    # Sanity check: required columns must exist
    required = [
        "Scenario",
        "Replications",
        "avg_spread_Mean", "avg_spread_CI_95",
        "mean_exec_time_Mean", "mean_exec_time_CI_95",
        "mm_final_pnl_per_1k_trades_Mean", "mm_final_pnl_per_1k_trades_CI_95",
        "price_volatility_Mean", "price_volatility_CI_95",
        "depth_within_5_ticks_Mean", "depth_within_5_ticks_CI_95",
        "mm_final_inventory_Mean", "mm_final_inventory_CI_95",
        "mm_realized_spread_Mean", "mm_realized_spread_CI_95",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in source file: {missing}")

    out = pd.DataFrame()
    out["Scenario"] = df["Scenario"]
    out["n_rep"] = df["Replications"]

    # Helper to expand Mean + CI_95 -> mean, lower, upper
    def add_metric(prefix, mean_col, ci_col):
        out[f"{prefix}_mean"] = df[mean_col]
        out[f"{prefix}_CI95_lower"] = df[mean_col] - df[ci_col]
        out[f"{prefix}_CI95_upper"] = df[mean_col] + df[ci_col]

    # Core metrics
    add_metric("avg_spread", "avg_spread_Mean", "avg_spread_CI_95")
    add_metric("mean_exec_time", "mean_exec_time_Mean", "mean_exec_time_CI_95")
    add_metric(
        "mm_final_pnl_per_1k_trades",
        "mm_final_pnl_per_1k_trades_Mean",
        "mm_final_pnl_per_1k_trades_CI_95",
    )

    # If you still care about these:
    add_metric("price_volatility", "price_volatility_Mean", "price_volatility_CI_95")
    add_metric("depth_within_5_ticks", "depth_within_5_ticks_Mean", "depth_within_5_ticks_CI_95")
    add_metric("mm_final_inventory", "mm_final_inventory_Mean", "mm_final_inventory_CI_95")
    add_metric("mm_realized_spread", "mm_realized_spread_Mean", "mm_realized_spread_CI_95")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {out.shape}")

if __name__ == "__main__":
    main()
