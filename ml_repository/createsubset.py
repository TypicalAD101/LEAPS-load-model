import pandas as pd
from datetime import timedelta
import os

# ==== INPUT FILE ====
input_path = "Actual_25_15_81_05_2006_FL TEST FOR IMPLEMENTATION.csv"   # <-- your file name

# ==== READ CSV ====
df = pd.read_csv(input_path)

# Try common timestamp column names
timestamp_cols = ["time", "timestamp", "date", "datetime", "Time", "Timestamp"]

ts_col = None
for col in timestamp_cols:
    if col in df.columns:
        ts_col = col
        break

if ts_col is None:
    raise ValueError("No timestamp column found. Add the name manually.")

# Convert timestamp to datetime
df[ts_col] = pd.to_datetime(df[ts_col])

# ==== SLICE FIRST 2 WEEKS ====
start_time = df[ts_col].min()
end_time = start_time + timedelta(days=14)

df_small = df[df[ts_col] <= end_time]

# ==== BUILD OUTPUT NAME ====
base, ext = os.path.splitext(input_path)
output_path = f"{base} smaller version{ext}"

# ==== SAVE ====
df_small.to_csv(output_path, index=False)

print("Smaller CSV created at:", output_path)
print("Original rows:", len(df))
print("New rows:", len(df_small))