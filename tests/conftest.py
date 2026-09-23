from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

for path in (str(SOURCE_ROOT), str(REPOSITORY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
