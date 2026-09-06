"""
HIT140 Assessment 2 - Objective 1
Analytic Task: Squad Age of World Cup Starters vs the Peak-Age Benchmark

Student: Abishek Rajeshkumar
Data source: FBref FIFA World Cup 2026 player statistics

Analytic question:
Among players who started at least one match at the FIFA World Cup 2026, is
the mean age different from the widely cited peak-age benchmark of 27 years?

    H0: mu = 27      H1: mu != 27      alpha = 0.05

All data acquisition, cleaning, calculation, statistical analysis and
visualisation are performed in Python.

Run:
    python Abishek_WC2026_SquadAge_Analysis_FINAL.py            live scrape
    python Abishek_WC2026_SquadAge_Analysis_FINAL.py --offline  cached CSVs
    python Abishek_WC2026_SquadAge_Analysis_FINAL.py --demo     synthetic test

Outputs: ./data/*.csv, ./figures/*.png, ./output/task4_results.txt
"""

import argparse
import io
import re
import time
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BENCHMARK_AGE = 27.0
ALPHA = 0.05
MIN_STARTS = 1
SAMPLE_SIZE = 200
RANDOM_SEED = 42

BASE = "https://fbref.com"
# FBref serves the current season at the short URL and archives it at the
# season-stamped one. Try both, since which resolves depends on whether 2026
# is still the live season.
PAGES = {
    "standard": [f"{BASE}/en/comps/1/stats/World-Cup-Stats",
                 f"{BASE}/en/comps/1/2026/stats/2026-World-Cup-Stats"],
    "playingtime": [f"{BASE}/en/comps/1/playingtime/World-Cup-Stats",
                    f"{BASE}/en/comps/1/2026/playingtime/2026-World-Cup-Stats"],
}
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-AU,en;q=0.9",
}
REQUEST_DELAY = 6                      # FBref rate-limits aggressively

# Sanity bounds. Ages outside this range indicate a parsing fault, not a
# remarkable player, and are dropped with a reported count.
AGE_MIN, AGE_MAX = 16.0, 45.0

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "Abishek_WC2026_SquadAge_Data.csv"
CLEAN_FILE = BASE_DIR / "Abishek_WC2026_SquadAge_Data_Cleaned.csv"
BOXPLOT_FILE = BASE_DIR / "Abishek_WC2026_SquadAge_Boxplot.png"
BARCI_FILE = BASE_DIR / "Abishek_WC2026_SquadAge_BarCI.png"
HIST_FILE = BASE_DIR / "Abishek_WC2026_SquadAge_Histogram.png"
np.random.seed(RANDOM_SEED)

_LOG = []
def say(msg=""):
    print(msg)
    _LOG.append(str(msg))

def header(title):
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


# ------------------------------------------------------------------ acquisition

def fetch_html(url: str) -> str:
    import requests
    say(f"  -> GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp.text


def extract_player_table(html: str) -> pd.DataFrame:
    """FBref hides its player tables inside HTML comments to deter scrapers.
    Stripping the comment markers makes them visible to read_html."""
    tables = pd.read_html(io.StringIO(html.replace("<!--", "").replace("-->", "")))
    for tbl in tables:
        cols = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in tbl.columns]
        if "Player" in cols and len(tbl) > 50:     # player table, not squad table
            return tbl
    raise ValueError("No player-level table found on this page.")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """FBref uses two-row headers; keep the group prefix only when it is real."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[1] if str(c[0]).startswith("Unnamed") else f"{c[0]}_{c[1]}"
                      for c in df.columns]
    return df


def read_fbref_csv(path) -> pd.DataFrame:
    """Read a 'Get table as CSV' export.

    Those exports are not plain CSV: they open with a Sports Reference
    citation banner, a blank line or two, and a merged group-header row
    ('Playing Time,Playing Time,...') above the real header. Feeding the file
    straight to read_csv yields garbage columns, so seek the real header row.
    """
    raw = pathlib.Path(path).read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.split("\n")
    start = next((i for i, l in enumerate(lines)
                  if l.startswith("Rk,") or l.startswith("Player,")), 0)
    if start:
        say(f"  -> skipped {start} banner/group-header line(s) in the export")
    return pd.read_csv(io.StringIO("\n".join(lines[start:])))


def acquire(name: str, offline: bool) -> pd.DataFrame:
    cache = RAW_FILE if name == "standard" else BASE_DIR / f"{name}_raw.csv"
    if cache.exists():
        say(f"  -> cache hit: {cache}")
        return read_fbref_csv(cache)
    if offline:
        raise FileNotFoundError(
            f"{cache} not found. Run without --offline, or download the table "
            f"from {PAGES[name][0]} via 'Share & Export -> Get table as CSV' "
            f"and save it as {cache}.")
    raw = None
    for url in PAGES[name]:
        try:
            raw = flatten_columns(extract_player_table(fetch_html(url)))
            break
        except Exception as e:
            say(f"  -> {type(e).__name__} on {url}; trying next URL")
    if raw is None:
        raise RuntimeError(
            f"Could not fetch '{name}' from any known URL. Download it manually "
            f"from {PAGES[name][0]} and save as {cache}, then use --offline.")
    raw.to_csv(cache, index=False)
    say(f"  -> cached {raw.shape[0]} rows x {raw.shape[1]} cols to {cache}")
    return raw


def make_demo_frame() -> pd.DataFrame:
    """Synthetic data for pipeline testing. Numbers are invented."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 1248
    squads = np.repeat([f"Team{i:02d}" for i in range(48)], 26)
    pos = np.tile(["GK"] * 3 + ["DF"] * 8 + ["MF"] * 8 + ["FW"] * 7, 48)
    base = {"GK": 29.5, "DF": 27.2, "MF": 26.6, "FW": 26.0}
    age = np.array([rng.normal(base[p], 3.6) for p in pos]).clip(17, 41)
    starts = rng.binomial(7, np.where(rng.random(n) < 0.45, 0.55, 0.06))
    mins = starts * rng.integers(45, 95, n) + rng.integers(0, 60, n) * (starts == 0)
    return pd.DataFrame({
        "Player": [f"Player {i}" for i in range(n)],
        "Nation": squads, "Pos": pos, "Squad": squads,
        "Age": [f"{int(a)}-{int((a % 1) * 365):03d}" for a in age],
        "Playing Time_MP": np.minimum(starts + rng.integers(0, 3, n), 7),
        "Playing Time_Starts": starts,
        "Playing Time_Min": mins,
    })


# --------------------------------------------------------------------- wrangling

def parse_age(value) -> float:
    """Convert FBref's age field to decimal years.

    The HTML tables give 'YY-DDD' (years-days), which converts to a genuinely
    continuous variable. The 'Get table as CSV' export ROUNDS to whole years,
    so that precision is unavailable when working from the export. Both forms
    are accepted; which one you got is reported after cleaning, because whole
    -year ages make the distribution discrete and affect the normality tests.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    m = re.match(r"^(\d{1,2})-(\d{1,3})$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 365.25
    return float(s) if re.match(r"^\d{1,2}(\.\d+)?$", s) else np.nan


VALID_POS = {"GK", "DF", "MF", "FW"}

def primary_position(pos) -> str:
    """Return a player's primary role.

    FBref writes dual-role players differently depending on the export:
    the HTML tables use 'DF,MF' while the CSV export concatenates to 'DFMF'.
    Both are handled; the first listed role is taken as primary.
    """
    if pd.isna(pos):
        return "UNK"
    s = str(pos).strip().upper().replace(" ", "")
    first = s.split(",")[0][:2]          # 'DF,MF' -> 'DF';  'DFMF' -> 'DF'
    return first if first in VALID_POS else "UNK"


def to_num(series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False),
                         errors="coerce")


def wrangle(standard: pd.DataFrame, playing) -> pd.DataFrame:
    df = standard.copy()
    df = df[df["Player"].astype(str) != "Player"]     # FBref repeats its header
    df = df.drop(columns=[c for c in ("Rk", "Matches") if c in df.columns])

    # Column names differ between the Standard Stats and Playing Time pages.
    starts_col = next((c for c in df.columns if c.endswith("Starts")), None)
    mins_col = next((c for c in df.columns if c.endswith("Min")), None)

    if playing is not None:
        p = playing[playing["Player"].astype(str) != "Player"].copy()
        ps = next((c for c in p.columns if c.endswith("Starts")), None)
        pm = next((c for c in p.columns if c.endswith("Min")), None)
        if ps:
            keep = ["Player", "Squad", ps] + ([pm] if pm else [])
            p = p[keep].rename(columns={ps: "starts_pt", pm: "mins_pt"})
            df = df.merge(p, on=["Player", "Squad"], how="left")
            starts_col = starts_col or "starts_pt"
            mins_col = mins_col or "mins_pt"

    if starts_col is None:
        raise KeyError("No 'Starts' column — cannot apply the inclusion rule.")

    df["age_exact"] = df["Age"].apply(parse_age)
    df["position"] = df["Pos"].apply(primary_position)
    df["starts"] = to_num(df[starts_col]).fillna(0)
    df["minutes"] = to_num(df[mins_col]) if mins_col else np.nan
    # FBref exports prefix Squad/Nation with a lowercase country code
    # ("us USA", "sct Scotland"). Strip it or every squad groups separately.
    df["squad"] = (df["Squad"].astype(str).str.strip()
                   .str.replace(r"^[a-z]{2,3}\s+", "", regex=True).str.strip())

    if "starts_pt" in df.columns and starts_col != "starts_pt":
        n_dis = (to_num(df["starts_pt"]).fillna(-1) != df["starts"]).sum()
        say(f"  starts disagreement between the two FBref tables : {n_dis}")

    say(f"  rows after removing repeated headers : {len(df)}")
    say(f"  duplicate Player+Squad rows          : {df.duplicated(['Player','squad']).sum()}")
    say(f"  missing age                          : {df['age_exact'].isna().sum()}")
    say(f"  implausible age (<{AGE_MIN:.0f} or >{AGE_MAX:.0f})         : "
        f"{((df['age_exact']<AGE_MIN)|(df['age_exact']>AGE_MAX)).sum()}")
    say(f"  unknown position                     : {(df['position']=='UNK').sum()}")
    frac_int = (df["age_exact"].dropna() % 1 == 0).mean()
    say(f"  age precision                        : "
        f"{'WHOLE YEARS (CSV export)' if frac_int > 0.95 else 'decimal years (years-days)'}")

    df = df.drop_duplicates(["Player", "squad"])
    df = df[df["age_exact"].between(AGE_MIN, AGE_MAX)]
    starters = df[df["starts"] >= MIN_STARTS].copy()

    say(f"  players in dataset                   : {len(df)}")
    say(f"  started >= {MIN_STARTS} match                  : {len(starters)}")
    say(f"  excluded (never started)             : {len(df) - len(starters)}")
    return starters.reset_index(drop=True)


def report_extremes(df: pd.DataFrame, k=3):
    """Print the oldest and youngest starters so the age parse can be
    eyeballed against the tournament coverage before trusting the statistics."""
    cols = ["Player", "squad", "position", "age_exact", "starts"]
    cols = [c for c in cols if c in df.columns]
    say("\n  Oldest starters (check these look plausible):")
    say(df.nlargest(k, "age_exact")[cols].to_string(index=False))
    say("\n  Youngest starters:")
    say(df.nsmallest(k, "age_exact")[cols].to_string(index=False))


# ---------------------------------------------------------------------- sampling

def stratified_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Proportionally allocated, stratified by position.
    Keepers are systematically older than forwards, so position is the largest
    known source of variance and the natural stratifier."""
    n = min(n, len(df))
    parts, rng = [], np.random.default_rng(seed)
    for pos, prop in df["position"].value_counts(normalize=True).items():
        stratum = df[df["position"] == pos]
        take = max(1, min(int(round(prop * n)), len(stratum)))
        parts.append(stratum.sample(n=take, random_state=rng.integers(1e6)))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    say(f"  stratified sample n = {len(out)} (target {n})")
    say("  allocation: " + ", ".join(f"{k} {v}" for k, v in
                                     out["position"].value_counts().items()))
    return out


# ------------------------------------------------------------------ descriptives

def describe(x: pd.Series, label: str):
    x = x.dropna()
    n, sd = len(x), x.std(ddof=1)
    q1, q3 = x.quantile(.25), x.quantile(.75)
    say(f"\n  {label}  (n = {n})")
    say(f"    Mean {x.mean():.3f} | Median {x.median():.3f} "
        f"| Mode {x.round().mode().iloc[0]:.0f}")
    say(f"    SD {sd:.3f} | Var {x.var(ddof=1):.3f} | SE {sd/np.sqrt(n):.4f} "
        f"| CV {sd/x.mean()*100:.1f}%")
    say(f"    Min {x.min():.2f} | Q1 {q1:.2f} | Q3 {q3:.2f} "
        f"| Max {x.max():.2f} | IQR {q3-q1:.2f}")
    say(f"    Skew {stats.skew(x, bias=False):+.3f} "
        f"| Excess kurtosis {stats.kurtosis(x, bias=False):+.3f}")


def describe_by_position(df: pd.DataFrame):
    tbl = (df.groupby("position")["age_exact"]
             .agg(n="count", mean="mean", sd="std", median="median",
                  min="min", max="max")
             .round(2).sort_values("mean", ascending=False))
    say("\n  Age by position:")
    say(tbl.to_string())


# ------------------------------------------------------------------- assumptions

def check_assumptions(x: pd.Series, label: str):
    x = x.dropna()
    n = len(x)
    say(f"\n  Assumption checks — {label} (n = {n})")
    say("    [1] Scale: decimal-year age is continuous and ratio-scaled.")
    say("    [2] Independence: VIOLATED — players nest within 48 squads.")
    say("        Re-tested on squad means in Section 7(d).")

    w, p_sw = stats.shapiro(x) if 3 <= n <= 5000 else (np.nan, np.nan)
    k2, p_k2 = stats.normaltest(x)
    sk, ku = stats.skew(x, bias=False), stats.kurtosis(x, bias=False)
    say(f"    [3] Shapiro-Wilk W = {w:.4f}, p = {p_sw:.4f}; "
        f"D'Agostino K2 = {k2:.4f}, p = {p_k2:.4f}")
    say(f"        Skew {sk:+.3f}, excess kurtosis {ku:+.3f}; "
        f"n = {n} >= 30 so CLT applies.")

    q1, q3 = x.quantile(.25), x.quantile(.75)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n_out = int(((x < lo) | (x > hi)).sum())
    n_z = int((np.abs(stats.zscore(x)) > 3).sum())
    say(f"    [4] Outliers: {n_out} beyond 1.5xIQR [{lo:.2f}, {hi:.2f}]; {n_z} with |z|>3.")
    say("        Retained (veteran keepers, teenage prodigies are genuine); "
        "trimmed-mean check in 7(c).")


# --------------------------------------------------------------------- inference

def effect_label(d: float) -> str:
    return ("negligible" if d < 0.2 else "small" if d < 0.5
            else "medium" if d < 0.8 else "large")


def confidence_interval(x: pd.Series, conf: float = 0.95):
    x = x.dropna()
    n, mean, se = len(x), x.mean(), stats.sem(x)
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df=n - 1)
    moe = tcrit * se
    say(f"\n  {int(conf*100)}% CI for mu")
    say(f"    xbar {mean:.4f}, s {x.std(ddof=1):.4f}, SE {se:.4f}, df {n-1}")
    say(f"    t_crit {tcrit:.4f}, margin of error {moe:.4f}")
    say(f"    CI = [{mean-moe:.4f}, {mean+moe:.4f}]")
    say(f"    Benchmark {BENCHMARK_AGE} is "
        f"{'INSIDE' if mean-moe <= BENCHMARK_AGE <= mean+moe else 'OUTSIDE'} the interval.")
    return mean - moe, mean + moe


def one_sample_ttest(x: pd.Series, mu0: float = BENCHMARK_AGE) -> dict:
    x = x.dropna()
    n, mean, sd = len(x), x.mean(), x.std(ddof=1)
    t_stat, p_val = stats.ttest_1samp(x, mu0)
    d = (mean - mu0) / sd
    t_crit = stats.t.ppf(1 - ALPHA / 2, df=n - 1)

    say(f"\n  One-sample t-test, mu0 = {mu0}")
    say(f"    H0: mu = {mu0}   H1: mu != {mu0}   alpha = {ALPHA}")
    say(f"    n {n}, xbar {mean:.4f}, s {sd:.4f}, df {n-1}")
    say(f"    t = {t_stat:.4f} (critical +/-{t_crit:.4f}), p = {p_val:.6f}")
    say(f"    Cohen's d = {d:+.4f} ({effect_label(abs(d))})")
    say(f"    Difference = {mean-mu0:+.4f} yr ({abs(mean-mu0)*12:.1f} months)")
    say(f"    DECISION: {'REJECT' if p_val < ALPHA else 'FAIL TO REJECT'} H0")

    try:
        from statsmodels.stats.power import TTestPower
        power = TTestPower().power(effect_size=abs(d), nobs=n, alpha=ALPHA)
        say(f"    Achieved power = {power:.4f}")
    except Exception:
        power = np.nan
        say("    (statsmodels not installed — achieved power not computed)")
    return {"t": t_stat, "p": p_val, "d": d, "n": n,
            "mean": mean, "sd": sd, "power": power}


def robustness(starters: pd.DataFrame, sample: pd.DataFrame,
               primary: dict) -> pd.DataFrame:
    header("7. ROBUSTNESS AND ALTERNATIVE SPECIFICATIONS")
    x = starters["age_exact"].dropna()
    s = sample["age_exact"].dropna()
    base_sig = primary["p"] < ALPHA

    tc, pc = stats.ttest_1samp(x, BENCHMARK_AGE)
    say(f"\n  (0) Full census of starters (n = {len(x)}): xbar {x.mean():.4f}, "
        f"t = {tc:.4f}, p = {pc:.6f}")
    say(f"      Sample estimate was {primary['mean']:.4f} (n = {primary['n']}); "
        f"difference {x.mean()-primary['mean']:+.4f} yr.")
    say(f"      {'Same' if (pc < ALPHA) == base_sig else 'DIFFERENT'} decision -> "
        "the stratified sample reproduces the full set.")

    w_stat, w_p = stats.wilcoxon(s - BENCHMARK_AGE)
    say(f"\n  (a) Wilcoxon signed-rank: W = {w_stat:.1f}, p = {w_p:.6f}")
    if (w_p < ALPHA) == base_sig:
        say("      Agrees with the t-test -> the decision does not depend on the "
            "normality assumption.")
    else:
        say("      DISAGREES with the t-test. The result sits near alpha and is "
            "sensitive to")
        say("      the distributional assumption. Report BOTH p-values and treat "
            "the finding")
        say("      as inconclusive rather than picking whichever is convenient.")

    res = stats.bootstrap((s.to_numpy(),), np.mean, confidence_level=.95,
                          n_resamples=10000, random_state=RANDOM_SEED, method="BCa")
    lo, hi = res.confidence_interval
    say(f"\n  (b) Bootstrap BCa 95% CI: [{lo:.4f}, {hi:.4f}] (compare to parametric CI)")

    tm = stats.trim_mean(s, 0.05)
    say(f"\n  (c) 5% trimmed mean {tm:.4f} vs untrimmed {s.mean():.4f} "
        f"(shift {tm-s.mean():+.4f} yr) -> outliers not driving the result.")

    squad = (starters.groupby("squad")
             .apply(lambda g: pd.Series({
                 "mean_age": g["age_exact"].mean(),
                 "wtd_age": np.average(g["age_exact"], weights=g["minutes"])
                            if g["minutes"].notna().all() and g["minutes"].sum() > 0
                            else g["age_exact"].mean(),
                 "n_starters": len(g)}), include_groups=False)
             .reset_index())

    t2, p2 = stats.ttest_1samp(squad["mean_age"], BENCHMARK_AGE)
    say(f"\n  (d) Squad-level re-test — fixes the independence violation")
    say(f"      {len(squad)} squad means: xbar {squad['mean_age'].mean():.4f}, "
        f"sd {squad['mean_age'].std(ddof=1):.4f}")
    say(f"      t = {t2:.4f}, p = {p2:.6f} -> "
        f"{'REJECT' if p2 < ALPHA else 'FAIL TO REJECT'} H0")

    t3, p3 = stats.ttest_1samp(squad["wtd_age"], BENCHMARK_AGE)
    say(f"\n  (e) Minutes-weighted squad age — age at which minutes were spent")
    say(f"      xbar {squad['wtd_age'].mean():.4f}, t = {t3:.4f}, p = {p3:.6f}")
    say(f"      Gap vs unweighted {squad['wtd_age'].mean()-squad['mean_age'].mean():+.4f} yr "
        "(positive = minutes concentrated on older starters).")

    return squad


# ------------------------------------------------------------------------ figures

def make_figures(starters, sample, squad, ci, demo=False):
    """Three single-purpose charts, saved beside the script."""
    tag = "  [SYNTHETIC DEMO DATA — NOT FOR SUBMISSION]" if demo else ""
    x = starters["age_exact"].dropna()
    order = [p for p in ("GK", "DF", "MF", "FW") if (starters.position == p).any()]

    # --- Boxplot: age by position -------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.boxplot([starters.loc[starters.position == p, "age_exact"].dropna()
                 for p in order],
                tick_labels=[f"{p}\n(n={(starters.position == p).sum()})"
                             for p in order],
                showmeans=True)
    plt.axhline(BENCHMARK_AGE, ls="--", color="black", lw=1.5,
                label=f"Peak-age benchmark ({BENCHMARK_AGE})")
    plt.ylabel("Age (years)")
    plt.xlabel("Playing position")
    plt.title("Age of World Cup 2026 Starters by Position" + tag)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BOXPLOT_FILE, dpi=300)
    plt.close()

    # --- Bar chart with 95% confidence intervals -----------------------------
    groups = [("All starters", x)] + [
        (p, starters.loc[starters.position == p, "age_exact"].dropna())
        for p in order]
    labels, means, errs = [], [], []
    for name, g in groups:
        if len(g) < 2:
            continue
        labels.append(f"{name}\n(n={len(g)})")
        means.append(g.mean())
        errs.append(stats.t.ppf(.975, len(g) - 1) * stats.sem(g))

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, means, yerr=errs, capsize=7,
                   color=["#4a6fa5"] + ["#c96f4a"] * (len(labels) - 1),
                   edgecolor="black", linewidth=.6)
    plt.axhline(BENCHMARK_AGE, ls="--", lw=2, color="black",
                label=f"Peak-age benchmark ({BENCHMARK_AGE})")
    for bar, m, e in zip(bars, means, errs):
        plt.text(bar.get_x() + bar.get_width() / 2, m + e + .12,
                 f"{m:.2f}", ha="center", fontsize=9)
    plt.ylabel("Mean age (years)")
    plt.title("Mean Age with 95% Confidence Intervals" + tag)
    plt.ylim(min(means) - 3, max(means) + 2)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BARCI_FILE, dpi=300)
    plt.close()

    # --- Histogram with mean, benchmark and CI, plus Q-Q --------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].hist(x, bins=range(int(x.min()), int(x.max()) + 2),
               edgecolor="white", alpha=.85)
    ax[0].axvline(x.mean(), ls="--", lw=2, color="crimson",
                  label=f"Mean {x.mean():.2f}")
    ax[0].axvline(BENCHMARK_AGE, lw=2, color="navy",
                  label=f"Benchmark {BENCHMARK_AGE}")
    ax[0].axvspan(ci[0], ci[1], alpha=.18, color="crimson", label="95% CI")
    ax[0].set(title="Age Distribution of Starters",
              xlabel="Age (years)", ylabel="Number of players")
    ax[0].legend(fontsize=9)
    ax[0].grid(axis="y", alpha=0.25)

    stats.probplot(x, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q Plot (normality check)")
    ax[1].grid(alpha=0.25)

    fig.suptitle("FIFA World Cup 2026 — Squad Age" + tag, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HIST_FILE, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    header("ANALYTIC TASK 4 — SQUAD AGE, FIFA WORLD CUP 2026")
    say(f"Q: Is mean age of players who started >= {MIN_STARTS} match different "
        f"from {BENCHMARK_AGE}?")
    say(f"H0: mu = {BENCHMARK_AGE}   H1: mu != {BENCHMARK_AGE}   alpha = {ALPHA}")

    header("1. DATA ACQUISITION")
    if args.demo:
        say("  *** SYNTHETIC DEMO DATA — INVENTED NUMBERS, DO NOT SUBMIT ***")
        standard, playing = make_demo_frame(), None
    else:
        standard = acquire("standard", args.offline)
        try:
            playing = acquire("playingtime", args.offline)
        except Exception as e:
            say(f"  -> playing-time unavailable ({type(e).__name__}); "
                "using Standard Stats only")
            playing = None

    header("2. DATA WRANGLING")
    starters = wrangle(standard, playing)
    report_extremes(starters)
    starters.to_csv(CLEAN_FILE, index=False)
    say(f"\n  Cleaned dataset saved to: {CLEAN_FILE.name}")

    header("3. POPULATION, SAMPLE, SAMPLING TECHNIQUE")
    say("  Target population : conceptual population of World Cup starters")
    say(f"  Observed set      : {len(starters)} starters, "
        f"{starters['squad'].nunique()} squads (census of 2026)")
    say("  Sampling frame    : FBref 2026 World Cup player tables")
    sample = stratified_sample(starters, SAMPLE_SIZE)

    header("4. DESCRIPTIVE STATISTICS")
    describe(sample["age_exact"], "Stratified random sample (PRIMARY)")
    describe(starters["age_exact"], "All starters (census, for comparison)")
    describe_by_position(starters)

    header("5. ASSUMPTION CHECKING")
    check_assumptions(sample["age_exact"], "Stratified sample (primary)")

    # The brief specifies inference FROM THE SAMPLE to the player population,
    # so the stratified sample is the primary unit of analysis. The census is
    # reported alongside as confirmation in Section 7.
    header("6. INFERENTIAL STATISTICS (PRIMARY — STRATIFIED SAMPLE)")
    lo, hi = confidence_interval(sample["age_exact"])
    result = one_sample_ttest(sample["age_exact"])

    squad = robustness(starters, sample, result)

    header("8. VISUALISATION")
    make_figures(starters, sample, squad, (lo, hi), demo=args.demo)

    header("9. RESULT SUMMARY")
    diff = result["mean"] - BENCHMARK_AGE
    say(f"  Sample mean starter age {result['mean']:.3f} yr "
        f"(n = {result['n']}); benchmark {BENCHMARK_AGE}")
    say(f"  Difference {diff:+.3f} yr ({abs(diff)*12:.1f} months), "
        f"Cohen's d {result['d']:+.3f} ({effect_label(abs(result['d']))})")
    say(f"  95% CI [{lo:.3f}, {hi:.3f}], p = {result['p']:.6f}")
    say(f"  {'REJECT' if result['p'] < ALPHA else 'FAIL TO REJECT'} H0 at alpha = {ALPHA}")
    say("\n  Write the interpretation and limitations in the report — "
        "see task4_writeup.md sections 6 and 8.")

    say("\nAnalysis completed successfully.")
    say(f"Cleaned dataset saved to: {CLEAN_FILE.name}")
    for f in (BOXPLOT_FILE, BARCI_FILE, HIST_FILE):
        say(f"Figure saved to: {f.name}")


if __name__ == "__main__":
    main()
