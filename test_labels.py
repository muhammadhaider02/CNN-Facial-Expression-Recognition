import pandas as pd

df = pd.read_parquet("metadata_train.parquet")

print("First 5 rows:")
print(df.head())

print("\nValence unique values:")
print(df["valence"].unique())

print("\nArousal unique values:")
print(df["arousal"].unique())

print("\nCounts of -2 in valence/arousal:")
print("Valence == -2:", (df["valence"] == -2).sum())
print("Arousal == -2:", (df["arousal"] == -2).sum())