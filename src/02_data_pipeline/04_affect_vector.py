from pathlib import Path
import pandas as pd






INPUT_PATH = Path(
    "data/processed/featured_interactions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/affect_interactions.csv"
)






print("Loading featured dataset...")

df = pd.read_csv(INPUT_PATH)

print("Rows:", len(df))






affect_columns = [
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)",
]






print("\n========== AFFECT COLUMN VALIDATION ==========")

for column in affect_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing affect column: {column}"
        )

    print(
        f"{column}: OK"
    )






print("\nCreating short affect feature names...")

df["affect_frustrated"] = (
    df["Average_confidence(FRUSTRATED)"]
)

df["affect_confused"] = (
    df["Average_confidence(CONFUSED)"]
)

df["affect_concentrating"] = (
    df["Average_confidence(CONCENTRATING)"]
)

df["affect_bored"] = (
    df["Average_confidence(BORED)"]
)






affect_features = [
    "affect_frustrated",
    "affect_confused",
    "affect_concentrating",
    "affect_bored",
]






print("\n========== AFFECT VALUE VALIDATION ==========")

for column in affect_features:

    print(f"\n{column}")

    print("Missing:", df[column].isna().sum())
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())
    print("Mean:", df[column].mean())






print("\n========== AFFECT SUM CHECK ==========")

affect_sum = df[affect_features].sum(axis=1)

print(
    "Rows approximately equal to 1:",
    ((affect_sum - 1).abs() < 1e-6).sum()
)

print(
    "Rows greater than 1:",
    (affect_sum > 1).sum()
)

print(
    "Maximum affect sum:",
    affect_sum.max()
)






print("\n========== AFFECT VECTOR ==========")

print(
    df[affect_features].head(10)
)






OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving affect-enhanced dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "Saved to:",
    OUTPUT_PATH
)

print(
    "Rows saved:",
    len(df)
)

print(
    "Columns saved:",
    len(df.columns)
)

print("\nAffect vector construction complete.")