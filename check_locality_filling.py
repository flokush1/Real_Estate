"""
Generate locality_filled.csv in artifact/data_transformation/
Columns: city, address_full, description, locality_original, locality_filled, source_of_fill
"""
import os
import pandas as pd
from real_estate.utils.locality_matcher import LocalityMatcher

MERGED_PATH = "artifact/data_ingestion/merged/merged.csv"
OUTPUT_DIR  = "artifact/data_transformation"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "locality_filled.csv")
JSON_PATH   = "real_estate_data/ncr_colonies.json"

df = pd.read_csv(MERGED_PATH)
print(f"Loaded {len(df)} rows.  Missing locality: {df['locality'].isna().sum()}")

matcher = LocalityMatcher(JSON_PATH)
print(f"Loaded {matcher.total_localities} localities across {len(matcher._city_choices)} cities\n")

locality_original = df["locality"].copy()

def fill_row(row):
    return matcher.extract_locality(
        city=row.get("city", ""),
        description=row.get("description", ""),
        address=row.get("address_full", ""),
        current_locality=row.get("locality", ""),
    )

df["locality_filled"] = df.apply(fill_row, axis=1)
df["locality_original"] = locality_original

def tag_source(row):
    orig   = row["locality_original"]
    filled = row["locality_filled"]
    if isinstance(orig, str) and orig.strip():
        return "original"
    if isinstance(filled, str) and filled.strip():
        return "filled"
    return "still_missing"

df["source_of_fill"] = df.apply(tag_source, axis=1)

out = df[["city", "address_full", "description",
          "locality_original", "locality_filled", "source_of_fill"]].copy()

os.makedirs(OUTPUT_DIR, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

originally   = (out["source_of_fill"] == "original").sum()
newly_filled = (out["source_of_fill"] == "filled").sum()
still_miss   = (out["source_of_fill"] == "still_missing").sum()

print(f"Saved  ->  {OUTPUT_PATH}")
print(f"Total rows       : {len(out)}")
print(f"Had locality     : {originally}")
print(f"Newly filled     : {newly_filled}  ({newly_filled / df['locality_original'].isna().sum() * 100:.1f}% of missing)")
print(f"Still missing    : {still_miss}")
print()
print("Per-city newly-filled breakdown:")
breakdown = (out[out["source_of_fill"] == "filled"]
             .groupby("city").size()
             .sort_values(ascending=False))
for city, cnt in breakdown.items():
    print(f"  {city:25s}: {cnt}")
