"""Repository entry point for the train/validation data checkpoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_recs.pipeline import main

if __name__ == "__main__":
    main()
