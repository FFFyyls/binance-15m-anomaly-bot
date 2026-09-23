import subprocess
import os
from pathlib import Path

def run_download():
    missing_file = "data/manifests/generated/missing_data.txt"
    if not os.path.exists(missing_file):
        print(f"{missing_file} not found")
        return

    data_dir = "data/raw/after_2026_02_01"
    script_path = "scripts/data/download_aggtrades_20s_rest.py"

    with open(missing_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        symbol, days_str = line.strip().split(":")
        days = days_str.split(",")

        # We can't easily pass multiple specific days to the existing script's main logic
        # without modifying it, but we can call it in a way that it only downloads what's missing
        # if we pass the symbol and correct date range.
        # However, the script's main() logic is tied to "months" or "end".
        # A better way is to call the underlying _download_symbol_days if we were to import it,
        # but since it's a CLI tool, let's use the CLI.

        # Given the script's logic:
        # if args.symbols_file is provided, it uses those symbols.
        # we can create a temporary symbols file for this one symbol.
        temp_sym_file = f"temp_sym_{symbol}.txt"
        with open(temp_sym_file, "w") as tf:
            tf.write(symbol)

        # Determine the min and max date for this symbol to cover all missing days
        sorted_days = sorted(days)
        start_date = sorted_days[0]
        end_date = sorted_days[-1]

        # Calculate how many months to cover this range (approx)
        # Or better, just modify a temporary version of the script or use it as is if it supports end date.
        # The script has --end and --months.
        # end_date is inclusive.

        # Actually, let's just use the CLI carefully for each symbol.
        # But wait, the script will try to download EVERYTHING in the range.
        # If the file already exists, it skips. So it's safe!

        print(f"Downloading missing data for {symbol} from {start_date} to {end_date}...")

        # The script calculates start_date = end_date - (months * 30 - 1)
        # We want start_date to be our earliest missing day.
        # so months = (end_date - start_date).days / 30 + 1
        d1 = dt_from_str(start_date)
        d2 = dt_from_str(end_date)
        delta_days = (d2 - d1).days + 1
        months = (delta_days // 30) + 1

        cmd = [
            "python3", script_path,
            "--symbols_file", temp_sym_file,
            "--data_dir", data_dir,
            "--end", end_date,
            "--months", str(months),
            "--workers", "4",
            "--sleep_s", "0.01"
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading for {symbol}: {e}")
        finally:
            if os.path.exists(temp_sym_file):
                os.remove(temp_sym_file)

def dt_from_str(s):
    import datetime
    return datetime.date.fromisoformat(s)

if __name__ == "__main__":
    run_download()
