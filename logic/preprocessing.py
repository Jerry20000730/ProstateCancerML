import pandas as pd
from util import util
import sys
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, src):
        self.src = src
        pass

    def preprocess(self):
        """
        Read the data
        """
        # check if the source file is an Excel file
        if not util.check_file_exists(self.src):
            print("[ERROR] The training source file '{}' does not exist.".format(self.src))
            sys.exit(1)
        if not util.is_xlsx_file(self.src):
            print("[ERROR] The training source file '{}' is not an Excel format file.".format(self.src))
            sys.exit(1)
        # Read the Excel file
        try:
            self.df = pd.read_excel(self.src)
            print("[INFO] Read the training source file '{}' successfully.".format(self.src))
        except Exception as e:
            print("[ERROR] Failed to read the training source file '{}': {}".format(self.src, e))
            sys.exit(1)
        
        # drop unnecessary columns
        self.df.drop(columns=["编号"], inplace=True)
        self.df.drop(columns=["身高"], inplace=True)
        self.df.drop(columns=["体重"], inplace=True)
        self.df.drop(columns=["bmi"], inplace=True)
        self.df.drop(columns=["PIRADS"], inplace=True)
        self.df.drop(columns=["载脂蛋白a1"], inplace=True)
        self.df.drop(columns=["碱性磷酸酶"], inplace=True)
        self.df.drop(columns=["载脂蛋白e"], inplace=True)
        # drop rows with any missing values
        missing_rows = self.df[self.df.isnull().any(axis=1)]
        print("[INFO] Rows with missing values (to be dropped):")
        print(missing_rows)
        self.df.dropna(inplace=True)

        # generate x and y
        # x is the features, y is the target variable
        new_df = self.df.copy()
        new_df["病理"] = self.df["病理"].map({"ca": 1, "nca": 0})
        self.y = new_df["病理"]
        self.df.drop(columns=["病理"], inplace=True)
        self.x = self.df.copy()

        # generate the features using test_and_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=40, stratify=self.y)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_x(self):
        """
        Get the features
        """
        if not hasattr(self, 'x'):
            print("[ERROR] The features have not been generated yet. Please run preprocess() first.")
            sys.exit(1)
        return self.x

    def get_y(self):
        """
        Get the target variable
        """
        if not hasattr(self, 'y'):
            print("[ERROR] The target variable has not been generated yet. Please run preprocess() first.")
            sys.exit(1)
        return self.y