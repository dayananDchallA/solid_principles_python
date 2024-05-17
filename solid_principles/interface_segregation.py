'''
To adhere to the Interface Segregation Principle (ISP) of the SOLID principles, interfaces should be designed to be small and client-specific rather than large and general-purpose. This means creating interfaces that provide only the methods that are of interest to the client.

Here’s how you can refactor your code to follow ISP:

1. Define separate interfaces for different types of processing tasks.
2. Ensure that each processing step class implements only the interfaces it needs.

Explanation:
-----------
1. Separate Interfaces: 
Define separate interfaces (Normalizer, Encoder, NaNFiller) for different types of processing tasks. Each interface has a single abstract method related to its specific task.
2. Specific Processing Step Classes: 
Each processing step class (NormalizeFeature, EncodeCategoricalFeature, FillNaNFeature) implements only the relevant interface.
3. process_data Function: 
This function takes separate lists of Normalizer, Encoder, and NaNFiller objects, ensuring that each type of processing step is applied only where appropriate.

By following ISP, each class is only responsible for a specific type of processing task, and the client (the process_data function) interacts with clearly defined interfaces, making the code more modular and easier to maintain.


'''

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod

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

# Separate interfaces for different processing tasks
class Normalizer(ABC):
    @abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class Encoder(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class NaNFiller(ABC):
    @abstractmethod
    def fill_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class NormalizeFeature(Normalizer):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        std = np.std(df[self.feature_name])
        mean = np.mean(df[self.feature_name])
        df[self.feature_name] = (df[self.feature_name] - mean) / std
        logging.info(f"Feature '{self.feature_name}' normalized")
        return df

class EncodeCategoricalFeature(Encoder):
    def __init__(self, feature_name: str, encoded_feature_name: str):
        self.feature_name = feature_name
        self.encoded_feature_name = encoded_feature_name
        self.encoder = LabelEncoder()

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.encoded_feature_name] = self.encoder.fit_transform(df[self.feature_name])
        logging.info(f"Feature '{self.feature_name}' encoded to '{self.encoded_feature_name}'")
        return df

class FillNaNFeature(NaNFiller):
    def __init__(self, feature_name: str, fill_value):
        self.feature_name = feature_name
        self.fill_value = fill_value

    def fill_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_name] = df[self.feature_name].fillna(self.fill_value)
        logging.info(f"NaN values in feature '{self.feature_name}' filled with '{self.fill_value}'")
        return df

def process_data(path: str, output_path: str, normalizers: list[Normalizer], encoders: list[Encoder], nan_fillers: list[NaNFiller]):
    # Load data
    data_loader = DataLoader(path)
    df = data_loader.load_data()

    # Apply normalization steps
    for normalizer in normalizers:
        df = normalizer.normalize(df)

    # Apply encoding steps
    for encoder in encoders:
        df = encoder.encode(df)

    # Apply NaN filling steps
    for nan_filler in nan_fillers:
        df = nan_filler.fill_nan(df)

    # Save processed data
    data_saver = DataSaver(output_path)
    data_saver.save_data(df)

def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    normalizers = [NormalizeFeature("feature_a")]
    encoders = [EncodeCategoricalFeature("feature_b", "feature_b_encoded")]
    nan_fillers = [FillNaNFeature("feature_c", -1)]
    process_data(path, output_path, normalizers, encoders, nan_fillers)

if __name__ == "__main__":
    main()
