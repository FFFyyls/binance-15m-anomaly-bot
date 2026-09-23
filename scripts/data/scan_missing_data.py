import os
import datetime as dt
from pathlib import Path

def get_missing_days(base_dir, start_date, end_date):
    start_dt = dt.date.fromisoformat(start_date)
    end_dt = dt.date.fromisoformat(end_date)

    all_days = []
    curr = start_dt
    while curr <= end_dt:
        all_days.append(curr.strftime("%Y-%m-%d"))
        curr += dt.timedelta(days=1)

    base_path = Path(base_dir)
    results = {}

    for sym_dir in base_path.iterdir():
        if not sym_dir.is_dir():
            continue

        symbol = sym_dir.name
        csv_dir = sym_dir / "20s"

        existing_days = set()
        if csv_dir.exists():
            for csv_file in csv_dir.glob("*.csv"):
                if csv_file.stat().st_size > 0:
                    existing_days.add(csv_file.stem)

        missing = [d for d in all_days if d not in existing_days]
        if missing:
            results[symbol] = missing

    return results

if __name__ == "__main__":
    base = "data/raw/after_2026_02_01"
    missing_map = get_missing_days(base, "2026-02-16", "2026-03-16")

    output = Path("data/manifests/generated/missing_data.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sym, days in sorted(missing_map.items()):
            f.write(f"{sym}:{','.join(days)}\n")

    print(f"Found {len(missing_map)} symbols with missing data. Saved to {output}")
