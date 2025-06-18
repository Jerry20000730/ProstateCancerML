import os
import pandas as pd

"""
check if a file exists at the given path.
:param filepath: The path to the file to check.
:return: True if the file exists, False otherwise.
"""
def check_file_exists(filepath:str) -> bool:
    return os.path.isfile(filepath)

"""
check if a file is xlsx format
"""
def is_xlsx_file(filepath:str) -> bool:
    return os.path.isfile(filepath) and filepath.lower().endswith(".xlsx")

"""
save metrics to excel
"""
def save_metrics_to_excel(metrics, filepath="output/result/metrics_summary.xlsx"):
    # automatically create the directory if it does not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_excel(filepath, index=False)
    print("[INFO] save metrics to: {}".format(filepath))