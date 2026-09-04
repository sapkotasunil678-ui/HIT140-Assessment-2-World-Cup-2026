"""
HIT140 Assessment 2 - Objective 1
Analytic Task: Goalkeeper Saves per 90 and Tournament Progression

Student: Sunil Sapkota
Data source: FBref FIFA World Cup 2026 goalkeeper statistics

Analytic question:
Is there a significant difference in average goalkeeper Saves/90 between
teams that reached the knockout stage and teams eliminated in the group stage?

All data cleaning, calculation, statistical analysis and visualisation are
performed in Python.
"""
from pathlib import Path

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 1. FILE LOCATION AND DATA IMPORT

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "Sunil_WC2026_Goalkeeping_Data.csv"

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"CSV file not found: {FILE_PATH}\n"
        "Place Sunil_WC2026_Goalkeeping_Data.csv in the same folder "
        "as this Python file."
    )
df = pd.read_csv(FILE_PATH)

# 2. DATA WRANGLING / CLEANING
required_columns = [
    "Goalkeeper",
    "Team",
    "Minutes",
    "90s",
    "Saves",
    "Progression",
]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {', '.join(missing_columns)}"
    )

# Clean text fields.
for column in ["Goalkeeper", "Team", "Progression"]:
    df[column] = df[column].astype(str).str.strip()
for column in ["Minutes", "90s", "Saves"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns).copy()

df = df[df["Minutes"] >= 90].copy()
df = df[df["90s"] > 0].copy()
df["Saves per 90"] = df["Saves"] / df["90s"]
valid_progressions = [
    "Reached knockout stage",
    "Eliminated in group stage",
]
df = df[df["Progression"].isin(valid_progressions)].copy()
# 3. DATA VALIDATION
expected_total = 58
expected_knockout = 39
expected_group = 19
knockout = df.loc[
    df["Progression"] == "Reached knockout stage",
    "Saves per 90",
].dropna()

group_eliminated = df.loc[
    df["Progression"] == "Eliminated in group stage",
    "Saves per 90",
].dropna()

if len(df) != expected_total:
    raise ValueError(
        f"Expected {expected_total} final records, but found {len(df)}."
    )

if len(knockout) != expected_knockout:
    raise ValueError(
        f"Expected {expected_knockout} knockout-stage records, "
        f"but found {len(knockout)}."
    )

if len(group_eliminated) != expected_group:
    raise ValueError(
        f"Expected {expected_group} group-stage records, "
        f"but found {len(group_eliminated)}."
    )

# 4. DESCRIPTIVE STATISTICS

def descriptive_statistics(values):
    return {
        "n": len(values),
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "minimum": values.min(),
        "median": values.median(),
        "maximum": values.max(),
    }


knockout_stats = descriptive_statistics(knockout)
group_stats = descriptive_statistics(group_eliminated)
# 5. 95% CONFIDENCE INTERVALS


def mean_confidence_interval(values, confidence=0.95):
    n = len(values)
    mean = values.mean()
    standard_error = stats.sem(values)

    lower, upper = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=standard_error,
    )

    return mean, lower, upper


knockout_mean, knockout_ci_low, knockout_ci_high = (
    mean_confidence_interval(knockout)
)

group_mean, group_ci_low, group_ci_high = (
    mean_confidence_interval(group_eliminated)
)
# 6. WELCH TWO-SAMPLE T-TEST

# H0: The population mean Saves/90 is the same in both groups.
# H1: The population mean Saves/90 is different between groups.

t_statistic, p_value = stats.ttest_ind(
    knockout,
    group_eliminated,
    equal_var=False,
)

alpha = 0.05

# 7. PRINT RESULTS

print("=" * 70)
print("HIT140 - GOALKEEPER SAVES PER 90 ANALYSIS")
print("=" * 70)

print("\nDATASET")
print(f"Final goalkeeper records: {len(df)}")
print(f"Knockout-stage records: {len(knockout)}")
print(f"Group-stage-eliminated records: {len(group_eliminated)}")

print("\nDESCRIPTIVE STATISTICS")
print("-" * 70)

print("\nReached knockout stage")
print(f"n       = {knockout_stats['n']}")
print(f"Mean    = {knockout_stats['mean']:.3f}")
print(f"SD      = {knockout_stats['std']:.3f}")
print(f"Minimum = {knockout_stats['minimum']:.3f}")
print(f"Median  = {knockout_stats['median']:.3f}")
print(f"Maximum = {knockout_stats['maximum']:.3f}")

print("\nEliminated in group stage")
print(f"n       = {group_stats['n']}")
print(f"Mean    = {group_stats['mean']:.3f}")
print(f"SD      = {group_stats['std']:.3f}")
print(f"Minimum = {group_stats['minimum']:.3f}")
print(f"Median  = {group_stats['median']:.3f}")
print(f"Maximum = {group_stats['maximum']:.3f}")

print("\n95% CONFIDENCE INTERVALS FOR THE MEAN")
print("-" * 70)
print(f"Knockout stage: {knockout_ci_low:.3f} to {knockout_ci_high:.3f}")
print(f"Group eliminated: {group_ci_low:.3f} to {group_ci_high:.3f}")

print("\nWELCH TWO-SAMPLE T-TEST")
print("-" * 70)
print(f"t-statistic = {t_statistic:.4f}")
print(f"p-value     = {p_value:.4f}")

if p_value < alpha:
    print(
        "Decision: Reject H0. There is statistically significant "
        "evidence of a difference in mean Saves/90."
    )
else:
    print(
        "Decision: Fail to reject H0. There is insufficient evidence "
        "of a difference in mean Saves/90."
    )

observed_difference = group_mean - knockout_mean

print("\nCONCLUSION")
print("-" * 70)
print(
    f"Observed difference in mean Saves/90 = {observed_difference:.3f}."
)
print(
    f"Group-stage-eliminated teams had the higher observed mean "
    f"({group_mean:.3f}) than knockout-stage teams ({knockout_mean:.3f})."
)
print(
    f"However, p = {p_value:.4f} > {alpha:.2f}, so the difference is "
    "not statistically significant at the 5% significance level."
)
print(
    "Therefore, the analysis does not provide sufficient evidence of a "
    "difference in population mean Saves/90 between the two groups."
)


# 8. SAVE CLEANED DATA

cleaned_file = BASE_DIR / "Sunil_WC2026_Goalkeeping_Data_Cleaned.csv"
df.to_csv(cleaned_file, index=False)

# 9. BOXPLOT

plot_file = BASE_DIR / "Sunil_WC2026_Goalkeeping_Saves90_Boxplot.png"

plt.figure(figsize=(8, 6))

plt.boxplot(
    [knockout, group_eliminated],
    tick_labels=["Knockout stage", "Group-stage eliminated"],
)

plt.ylabel("Saves per 90 minutes")
plt.xlabel("Tournament progression")
plt.title("Goalkeeper Saves per 90 by Tournament Progression")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.show()

print("\nAnalysis completed successfully.")
print(f"Box plot saved to: {plot_file}")
print(f"Cleaned dataset saved to: {cleaned_file}")
print("=" * 70)
