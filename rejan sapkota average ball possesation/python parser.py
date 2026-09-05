"""Extract the saved FBref squad table into csv."""
import csv
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SOURCE = "https://fbref.com/en/comps/1/possession/World-Cup-Stats"


class SquadTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.rows = []
        self.row = None
        self.key = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.active = attrs.get("id") == "stats_squads_possession_for"
        if not self.active:
            return
        if tag == "tr":
            self.row = {}
        if tag in ("td", "th"):
            self.key = attrs.get("data-stat")
            self.text = []

    def handle_data(self, data):
        if self.active and self.key:
            self.text.append(data)

    def handle_endtag(self, tag):
        if self.active:
            if tag in ("td", "th") and self.key:
                self.row[self.key] = "".join(self.text).strip()
                self.key = None
            if tag == "tr" and self.row is not None:
                if self.row.get("team") not in (None, "Squad"):
                    self.rows.append(self.row)
                self.row = None
        if tag == "table":
            self.active = False


def main():
    parser = SquadTable()
    parser.feed((DATA / "World-Cup-Stats.htm").read_text(encoding="utf-8"))
    if len(parser.rows) != 48:
        raise ValueError(f"Expected 48 teams; found {len(parser.rows)}")
    records = []
    for row in parser.rows:
        possession = float(row["possession"])
        if not 0 <= possession <= 100:
            raise ValueError("Possession outside 0-100")
        records.append({
            "Squad_As_Displayed": row["team"],
            "Average_Possession_Percent": possession,
            "Players_Used": int(row["players_used"]),
            "Minutes_90s": float(row["minutes_90s"]),
            "Scope": "Whole tournament (includes knockout matches)",
            "Source_URL": SOURCE,
        })
    output = DATA / "Rejan_FBref_Whole_Tournament_Possession.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} teams to {output.name}")
    print("This table does not provide group-stage-only possession or verified advancement labels.")


if __name__ == "__main__":
    main()
