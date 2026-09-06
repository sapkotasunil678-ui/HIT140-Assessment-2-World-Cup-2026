"""Compare whole-tournament possession by advancement."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"
VALUE = "Average_Possession_Percent"
GROUPS = ["Advanced", "Eliminated"]
FIFA_SOURCE = "https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub-knockout-stage.php"


def main():
    # Load data and label advancement.
    df = pd.read_csv(DATA / "cleaned.csv")
    fixtures = pd.read_csv(DATA / "round32.csv")
    if len(df) != 48 or df.Team.isna().any() or df.Team.duplicated().any():
        raise ValueError("Expected 48 unique teams with non-missing names.")
    df[VALUE] = pd.to_numeric(df[VALUE], errors="coerce")
    if not df[VALUE].between(0, 100).all():
        raise ValueError("Missing or invalid possession percentage.")
    if not df.Scope.eq("Whole tournament (includes knockout matches)").all():
        raise ValueError("Unexpected scope: review the source data.")
    qualified = pd.concat([fixtures.Team1, fixtures.Team2]).replace(
        {"Bosnia and Herzegovina": "Bosnia–Herz"})
    if len(fixtures) != 16 or qualified.isna().any() or qualified.nunique() != 32:
        raise ValueError("Expected 16 round-of-32 fixtures with 32 unique teams.")
    if not set(qualified).issubset(set(df.Team)):
        raise ValueError("A FIFA team name does not match the possession dataset.")
    df["Progression"] = np.where(df.Team.isin(qualified), "Advanced", "Eliminated")
    df["Progression_Source_URL"] = FIFA_SOURCE
    if df.Progression.value_counts().to_dict() != {"Advanced": 32, "Eliminated": 16}:
        raise ValueError("Incorrect advancement counts.")

    # Sample 75% per group without replacement; fixed seeds.
    selected = []
    for group, n, seed in [("Advanced", 24, 140), ("Eliminated", 12, 141)]:
        names = sorted(df.loc[df.Progression.eq(group), "Team"])
        selected.extend(random.Random(seed).sample(names, n))
    sample = df.loc[df.Team.isin(selected)].sort_values("Team").copy()
    a = sample.loc[sample.Progression.eq("Advanced"), VALUE]
    b = sample.loc[sample.Progression.eq("Eliminated"), VALUE]

    # Descriptive statistics and 95% mean intervals.
    full_summary = df.groupby("Progression")[VALUE].describe()
    summary = sample.groupby("Progression")[VALUE].describe()
    for group, values in zip(GROUPS, [a, b]):
        low, high = stats.t.interval(.95, len(values)-1,
                                    loc=values.mean(), scale=stats.sem(values))
        summary.loc[group, "CI95_Lower"] = low
        summary.loc[group, "CI95_Upper"] = high

    # One-sided Welch t-test.
    # H0: mu_advanced <= mu_eliminated; H1: mu_advanced > mu_eliminated.
    alpha = .05
    test = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    if not np.isfinite(test.statistic) or not np.isfinite(test.pvalue):
        raise ValueError("The sample does not support a finite Welch test.")
    difference = a.mean() - b.mean()
    # Two-sided 95% difference interval.
    two_sided = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    difference_ci = two_sided.confidence_interval(confidence_level=.95)
    # Lower bound matching the one-sided test.
    directional_ci = test.confidence_interval(confidence_level=.95)
    decision = (
        "Reject H0: evidence of higher average possession among advancing teams, under the test assumptions."
        if test.pvalue < alpha else
        "Fail to reject H0: insufficient evidence of higher average possession among advancing teams."
    )

    # Save CSV outputs.
    OUT.mkdir(exist_ok=True)
    df.to_csv(DATA / "analysis_data.csv", index=False)
    sample.to_csv(DATA / "sample.csv", index=False)
    full_summary.to_csv(OUT / "all_teams_statistics.csv")
    summary.to_csv(OUT / "sample_statistics.csv")
    pd.DataFrame([{"Difference_pp": difference, "t": test.statistic, "df": test.df,
                   "One_sided_p": test.pvalue, "CI95_Lower": difference_ci.low,
                   "CI95_Upper": difference_ci.high,
                   "One_sided_95_Lower": directional_ci.low}]).to_csv(OUT / "ttest.csv", index=False)
    report = "\n".join([
        "REJAN SAPKOTA — POSSESSION ANALYSIS",
        "Question: Did advancing teams have higher whole-tournament average possession?",
        "\nALL 48 TEAMS (census; descriptive comparison):", full_summary.round(4).to_string(),
        "\nSTRATIFIED RANDOM SAMPLE: 24 advanced, 12 eliminated; seeds 140 and 141.",
        "Each observation is a team's published whole-tournament possession average.",
        "SAMPLE STATISTICS AND MODEL-BASED 95% t INTERVALS:", summary.round(4).to_string(),
        "\nH0: mu_advanced <= mu_eliminated. H1: mu_advanced > mu_eliminated.",
        f"One-sided Welch two-sample t-test; alpha = {alpha}.",
        f"Mean difference: {difference:.4f} percentage points (Advanced - Eliminated).",
        f"t = {test.statistic:.4f}; df = {test.df:.4f}; p = {test.pvalue:.6f}.",
        f"Two-sided 95% difference CI: [{difference_ci.low:.4f}, {difference_ci.high:.4f}] percentage points.",
        f"One-sided 95% difference CI: [{directional_ci.low:.4f}, infinity) percentage points.",
        decision,
        "\nLIMITATIONS:",
        "Whole-tournament averages include extra knockout matches for advancing teams.",
        "These results cannot isolate group-stage possession or establish causation.",
        "Welch assumes independent observations and suitable sampling distributions; teams share opponents.",
        "The small eliminated sample is right-skewed; normality and independence are imperfect approximations.",
        "The 48-team dataset is a census. Sampling is included to demonstrate the assignment skill.",
        "Conventional t intervals have no finite-population correction despite a 75% sampling fraction.",
        "They are model-based teaching calculations, not exact design-based inference for the finite tournament.",
        "Results do not automatically generalise to other tournaments.",
        "\nSOURCES:", str(df.Source_URL.iloc[0]), FIFA_SOURCE,
    ])
    print(report)

    # Boxplot and team values.
    labels = ["Advanced (n=24)", "Eliminated (n=12)"]
    colors = ["#3979A8", "#D99532"]
    fig, ax = plt.subplots(figsize=(8, 5))
    boxes = ax.boxplot([a, b], tick_labels=labels, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(.4)
    for position, values, color in zip([1, 2], [a, b], colors):
        ax.scatter(np.linspace(position-.08, position+.08, len(values)),
                   sorted(values), s=20, color=color)
    ax.set(title="Possession by progression — sampled teams",
           ylabel="Whole-tournament average possession (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "boxplot.png", dpi=200)
    plt.close(fig)

    # Mean possession with 95% intervals.
    means = summary.loc[GROUPS, "mean"].to_numpy()
    errors = np.vstack([means-summary.loc[GROUPS, "CI95_Lower"].to_numpy(),
                        summary.loc[GROUPS, "CI95_Upper"].to_numpy()-means])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=errors, capsize=8, width=.55, color=colors)
    ax.set(title="Sample mean possession with 95% t confidence intervals",
           ylabel="Whole-tournament average possession (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "mean_ci.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
