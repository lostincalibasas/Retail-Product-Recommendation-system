"""Materialize and evaluate the approved final test once, without training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from retail_recs.heldout import main

if __name__ == "__main__":
    main()
