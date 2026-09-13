import pandas as pd

INPUT_FILE = "data/processed/clean_interactions.csv"

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")

# ---------------------------------------------------------
# 1. Basic problem/skill availability
# ---------------------------------------------------------

print("\n--- Problem / Skill availability ---")

total_rows = len(df)

known_skill_rows = df["skill_id"].notna().sum()
missing_skill_rows = df["skill_id"].isna().sum()

print(f"Rows with known skill_id:   {known_skill_rows:,}")
print(f"Rows with missing skill_id: {missing_skill_rows:,}")

print(f"Known skill percentage:   {known_skill_rows / total_rows * 100:.2f}%")
print(f"Missing skill percentage: {missing_skill_rows / total_rows * 100:.2f}%")


# ---------------------------------------------------------
# 2. Build problem -> skill mapping
# ---------------------------------------------------------

print("\n--- Building problem -> skill mapping ---")

# Ignore rows where problem_id itself is missing
problem_df = df[df["problem_id"].notna()].copy()

problem_stats = (
    problem_df
    .groupby("problem_id")
    .agg(
        total_interactions=("problem_id", "size"),
        known_skill_interactions=("skill_id", lambda x: x.notna().sum()),
        missing_skill_interactions=("skill_id", lambda x: x.isna().sum()),
        unique_known_skills=("skill_id", "nunique")
    )
    .reset_index()
)

problem_stats["known_skill_percentage"] = (
    problem_stats["known_skill_interactions"]
    / problem_stats["total_interactions"]
    * 100
)

print(f"Unique problems: {len(problem_stats):,}")


# ---------------------------------------------------------
# 3. How many problems have completely missing skills?
# ---------------------------------------------------------

print("\n--- Problem-level skill availability ---")

completely_missing = (
    problem_stats["known_skill_interactions"] == 0
).sum()

completely_known = (
    problem_stats["missing_skill_interactions"] == 0
).sum()

partially_known = (
    (problem_stats["known_skill_interactions"] > 0)
    &
    (problem_stats["missing_skill_interactions"] > 0)
).sum()

print(f"Problems with skill known for ALL interactions:      {completely_known:,}")
print(f"Problems with skill known for SOME interactions:      {partially_known:,}")
print(f"Problems with skill missing for ALL interactions:     {completely_missing:,}")


# ---------------------------------------------------------
# 4. Interactions that may be recoverable
# ---------------------------------------------------------

print("\n--- Potentially recoverable interactions ---")

recoverable_interactions = (
    problem_df["problem_id"].isin(
        problem_stats.loc[
            problem_stats["known_skill_interactions"] > 0,
            "problem_id"
        ]
    )
    &
    problem_df["skill_id"].isna()
).sum()

print(
    "Missing-skill interactions belonging to a problem "
    f"that has a known skill elsewhere: {recoverable_interactions:,}"
)

print(
    f"Percentage of missing-skill interactions potentially recoverable: "
    f"{recoverable_interactions / missing_skill_rows * 100:.2f}%"
)


# ---------------------------------------------------------
# 5. Check whether a problem maps to multiple skills
# ---------------------------------------------------------

print("\n--- Problem -> Skill consistency ---")

multiple_skill_problems = (
    problem_stats["unique_known_skills"] > 1
).sum()

print(
    f"Problems associated with multiple known skills: "
    f"{multiple_skill_problems:,}"
)

print(
    f"Percentage of problems with multiple known skills: "
    f"{multiple_skill_problems / len(problem_stats) * 100:.2f}%"
)


# ---------------------------------------------------------
# 6. Show examples
# ---------------------------------------------------------

print("\n--- Examples of potentially recoverable problems ---")

examples = (
    problem_stats[
        (problem_stats["known_skill_interactions"] > 0)
        &
        (problem_stats["missing_skill_interactions"] > 0)
    ]
    .sort_values(
        ["missing_skill_interactions", "total_interactions"],
        ascending=False
    )
    .head(10)
)

print(examples.to_string(index=False))


# ---------------------------------------------------------
# 7. Final interpretation
# ---------------------------------------------------------

print("\n--- Investigation complete ---")