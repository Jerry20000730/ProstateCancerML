import os

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