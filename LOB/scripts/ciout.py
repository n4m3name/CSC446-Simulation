import pandas as pd

def main():
    in_path = "data/lob_y_CI95_by_scenario.csv"

    df = pd.read_csv(in_path)

    # Define the three metrics of interest and their CI columns
    metrics = {
        "avg_spread": {
            "lower": "avg_spread_CI95_lower",
            "upper": "avg_spread_CI95_upper",
            "outfile": "appendix_B_avg_spread_CI.tex"
        },
        "median_exec_time": {
            "lower": "median_exec_time_CI95_lower",
            "upper": "median_exec_time_CI95_upper",
            "outfile": "appendix_B_median_exec_time_CI.tex"
        },
        "mm_final_pnl_per_1k_trades": {
            "lower": "mm_final_pnl_per_1k_trades_CI95_lower",
            "upper": "mm_final_pnl_per_1k_trades_CI95_upper",
            "outfile": "appendix_B_mm_pnl_CI.tex"
        },
    }

    for metric_name, spec in metrics.items():
        lower_col = spec["lower"]
        upper_col = spec["upper"]
        outfile = spec["outfile"]

        # Build a compact DataFrame for this metric only
        metric_df = df[["Scenario", lower_col, upper_col]].copy()

        # Rename columns to avoid underscores (so LaTeX treats them as text, not math)
        metric_df.columns = ["Scenario", "Lower 95% CI", "Upper 95% CI"]

        # Convert to LaTeX. escape=True (default) keeps things as text.
        latex_tabular = metric_df.to_latex(index=False, float_format="%.6g")

        # Write just the tabular environment to a .tex file
        with open(outfile, "w") as f:
            f.write(latex_tabular)

if __name__ == "__main__":
    main()
