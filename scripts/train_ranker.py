"""Repository entry point for frozen-candidate model training and validation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retail_recs.train_ranker import main

if __name__ == "__main__":
    main()
