import pandas as pd

def main():
    in_path = "data/lob_y_CI95_by_scenario.csv"
    out_path = "data/appendix_CI_primary_metrics.tex"

    df = pd.read_csv(in_path)

    # Columns to keep: Scenario + CI bounds for the 3 primary metrics (no fill rate)
    cols = [
        "Scenario",
        "avg_spread_CI95_lower",
        "avg_spread_CI95_upper",
        "price_volatility_CI95_lower",
        "price_volatility_CI95_upper",
        "mm_final_pnl_per_1k_trades_CI95_lower",
        "mm_final_pnl_per_1k_trades_CI95_upper",
    ]

    ci_df = df[cols]

    # Convert to LaTeX tabular; adjust float_format if you want more/less precision
    latex_table = ci_df.to_latex(index=False, float_format="%.6g")

    # Write directly to a .tex file for inclusion in the appendix
    with open(out_path, "w") as f:
        f.write(latex_table)

if __name__ == "__main__":
    main()
