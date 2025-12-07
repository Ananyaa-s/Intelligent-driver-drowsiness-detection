# check_rows.py
import pandas as pd

data = pd.read_csv("user_data.csv")

if data.empty:
    print("❌ CSV is empty.")
else:
    print(f"✅ CSV has {len(data)} rows.")
    print(data.head())
