import pandas as pd

def main():
    in_path = "data/lob_y_CI95_by_scenario.csv"
    out_path = "data/appendix_CI_table.tex"

    df = pd.read_csv(in_path)

    # Select only CI columns + scenario label
    ci_cols = ["Scenario"] + sorted(
        [
            c for c in df.columns
            if c.endswith("_CI95_lower") or c.endswith("_CI95_upper")
        ]
    )
    ci_df = df[ci_cols]

    # Convert to LaTeX
    latex_table = ci_df.to_latex(index=False, float_format="%.6g")

    # Write to .tex file
    with open(out_path, "w") as f:
        f.write(latex_table)

if __name__ == "__main__":
    main()
