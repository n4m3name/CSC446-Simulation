import os

INPUT_FILE = "data/simulation_results_full.csv"
OUTPUT_DIR = "data/chunks"
LINES_PER_FILE = 100

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    header = f.readline()        # Read header
    lines = f.readlines()        # Read the rest

# --- If you want every output file to include the header, keep this True ---
INCLUDE_HEADER = True

# Split into chunks of N lines
for i in range(0, len(lines), LINES_PER_FILE):
    chunk = lines[i : i + LINES_PER_FILE]
    part_num = i // LINES_PER_FILE + 1
    output_path = os.path.join(OUTPUT_DIR, f"part_{part_num:03d}.csv")

    with open(output_path, "w", encoding="utf-8") as out:
        if INCLUDE_HEADER:
            out.write(header)
        out.writelines(chunk)

print("Done.")
