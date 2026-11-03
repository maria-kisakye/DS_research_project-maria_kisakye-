
# from pathlib import Path
# import pandas as pd

# # Locate the project folder
# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# # Locate the raw dataset folder
# DATA_DIR = PROJECT_ROOT / "data" / "raw"

# # Read the three datasets
# performance = pd.read_csv(DATA_DIR / "trainperf.csv")
# demographics = pd.read_csv(DATA_DIR / "traindemographics.csv")
# previous_loans = pd.read_csv(DATA_DIR / "trainprevloans.csv")

# # Display the number of records
# print("Loan performance records:", len(performance))
# print("Demographic records:", len(demographics))
# print("Previous loan records:", len(previous_loans))

# # Display the columns in each dataset
# for name, dataframe in [
#     ("Performance", performance),
#     ("Demographics", demographics),
#     ("Previous Loans", previous_loans),
# ]:
#     print(f"\n{name} columns:")
#     print(dataframe.columns.tolist())

#     print(f"\n{name} missing values:")
#     print(dataframe.isnull().sum())
