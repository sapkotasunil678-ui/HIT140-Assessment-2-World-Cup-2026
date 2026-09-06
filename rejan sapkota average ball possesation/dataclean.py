"""Clean the extracted CSV using Python's standard library.

Keep whole-tournament scope. Do not invent advancement labels or missing values.
Run this file from Python, Anaconda Prompt, or a Jupyter notebook with %run.
"""
import csv
import math
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SOURCE = DATA / "data.csv"
OUTPUT = DATA / "cleaned.csv"


def clean():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = ["Squad_As_Displayed", "Average_Possession_Percent",
                   "Players_Used", "Minutes_90s", "Scope", "Source_URL"]
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError("Required columns are missing from the source CSV.")
        raw = list(reader)

    cleaned, seen_rows, seen_teams = [], set(), set()
    duplicates = 0
    for line, row in enumerate(raw, start=2):
        # Strip surrounding spaces without changing the original source file.
        row = {key: (row.get(key) or "").strip() for key in columns}
        if any(not value for value in row.values()):
            raise ValueError(f"Missing data on line {line}; check the source webpage.")

        # Remove only exact duplicate records, never conflicting team records.
        identity = tuple(row[key] for key in columns)
        if identity in seen_rows:
            duplicates += 1
            continue
        seen_rows.add(identity)

        # FBref displays a flag code before the team name, e.g. 'dz Algeria'.
        # Preserve the original label and store its code separately.
        match = re.fullmatch(r"([a-z]{2,3})\s+(.+)", row["Squad_As_Displayed"])
        if not match:
            raise ValueError(f"Unexpected team label on line {line}.")
        code, team = match.groups()
        team = " ".join(team.split())
        if team.casefold() in seen_teams:
            raise ValueError(f"Conflicting records for {team}; check before proceeding.")
        seen_teams.add(team.casefold())

        possession = float(row["Average_Possession_Percent"].removesuffix("%").strip())
        players = float(row["Players_Used"])
        minutes_90s = float(row["Minutes_90s"])
        if not math.isfinite(possession) or not 0 <= possession <= 100:
            raise ValueError(f"Invalid possession for {team}.")
        if not math.isfinite(players) or players <= 0 or not players.is_integer():
            raise ValueError(f"Invalid player count for {team}.")
        if not math.isfinite(minutes_90s) or minutes_90s <= 0:
            raise ValueError(f"Invalid 90-minute equivalents for {team}.")
        cleaned.append({"Team": team, "FBref_Flag_Code": code,
                        "Average_Possession_Percent": possession,
                        "Players_Used": int(players), "Minutes_90s": minutes_90s,
                        "Squad_As_Displayed": row["Squad_As_Displayed"],
                        "Scope": row["Scope"], "Source_URL": row["Source_URL"]})

    if len(cleaned) != 48:
        raise ValueError(f"Expected 48 unique teams; found {len(cleaned)}.")
    cleaned.sort(key=lambda row: row["Team"].casefold())
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cleaned[0]))
        writer.writeheader()
        writer.writerows(cleaned)
    print(f"Input rows: {len(raw)}")
    print(f"Exact duplicates removed: {duplicates}")
    print(f"Cleaned teams: {len(cleaned)}")
    print("Missing values and numeric validation: passed")
    print(f"Saved: {OUTPUT.name}")
    print("Scope remains whole tournament. Advancement labels are still required.")


if __name__ == "__main__":
    clean()
