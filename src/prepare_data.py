
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

def load_data():
    """Load the original Nigerian loan datasets."""
    performance = pd.read_csv(DATA_DIR / "trainperf.csv")
    demographics = pd.read_csv(DATA_DIR / "traindemographics.csv")
    previous_loans = pd.read_csv(DATA_DIR / "trainprevloans.csv")

    return performance, demographics, previous_loans


if __name__ == "__main__":
    performance, demographics, previous_loans = load_data()

    print("Performance:", performance.shape)
    print("Demographics:", demographics.shape)
    print("Previous loans:", previous_loans.shape)
