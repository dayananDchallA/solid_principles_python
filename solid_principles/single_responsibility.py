'''
To adhere to the Single Responsibility Principle (SRP) of the SOLID principles, each class or function should have only one responsibility or reason to change. This makes your code easier to understand, maintain, and extend.

Here’s how you can refactor your code to follow SRP:

1. Separate the logic for reading data, processing data, and writing data into distinct functions or classes.
2. Create individual classes or functions for each processing step.

Explanation:
-----------
1. DataLoader Class: Responsible for loading data from a specified path.
2. DataSaver Class: Responsible for saving data to a specified output path.
3. NormalizeFeature Class: Responsible for normalizing a specified feature.
4. EncodeCategoricalFeature Class: Responsible for encoding a specified categorical feature.
5. FillNaNFeature Class: Responsible for filling NaN values in a specified feature.
6. process_data Function: Orchestrates the loading, processing, and saving of data by using the above classes.
7. main Function: Specifies the paths and calls the process_data function.

Each class and function now has a single responsibility, making the code easier to understand, test, and maintain.

'''
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        logging.info(f"Data loaded from {self.path}: {df.head()}")
        return df

class DataSaver:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def save_data(self, df: pd.DataFrame):
        df.to_parquet(self.output_path)
        logging.info(f"Data saved to {self.output_path}")

class NormalizeFeature:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        std = np.std(df[self.feature_name])
        mean = np.mean(df[self.feature_name])
        df[self.feature_name] = (df[self.feature_name] - mean) / std
        logging.info(f"Feature '{self.feature_name}' normalized")
        return df

class EncodeCategoricalFeature:
    def __init__(self, feature_name: str, encoded_feature_name: str):
        self.feature_name = feature_name
        self.encoded_feature_name = encoded_feature_name
        self.encoder = LabelEncoder()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.encoded_feature_name] = self.encoder.fit_transform(df[self.feature_name])
        logging.info(f"Feature '{self.feature_name}' encoded to '{self.encoded_feature_name}'")
        return df

class FillNaNFeature:
    def __init__(self, feature_name: str, fill_value):
        self.feature_name = feature_name
        self.fill_value = fill_value

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_name] = df[self.feature_name].fillna(self.fill_value)
        logging.info(f"NaN values in feature '{self.feature_name}' filled with '{self.fill_value}'")
        return df

def process_data(path: str, output_path: str):
    # Load data
    data_loader = DataLoader(path)
    df = data_loader.load_data()

    # Apply processing steps
    df = NormalizeFeature("feature_a").apply(df)
    df = EncodeCategoricalFeature("feature_b", "feature_b_encoded").apply(df)
    df = FillNaNFeature("feature_c", -1).apply(df)

    # Save processed data
    data_saver = DataSaver(output_path)
    data_saver.save_data(df)

def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    process_data(path, output_path)

if __name__ == "__main__":
    main()
