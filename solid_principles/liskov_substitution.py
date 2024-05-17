'''
To adhere to the Liskov Substitution Principle (LSP) of the SOLID principles, subclasses should be substitutable for their base classes without affecting the correctness of the program. This means that any instance of a subclass should be able to replace an instance of the base class without altering the desirable properties of the program.

Here’s how you can refactor your code to follow LSP:

1. Define a base class for data processing steps.
2. Ensure all specific processing step classes inherit from this base class and implement its methods.
3. Use the base class type to define the expected interface in functions that apply processing steps.

Explanation:
-----------
1. DataProcessingStep Base Class: 
This is an abstract base class with an abstract method apply. This method must be implemented by all subclasses.
2. Specific Processing Steps: 
NormalizeFeature, EncodeCategoricalFeature, and FillNaNFeature are concrete classes that inherit from DataProcessingStep and implement the apply method.
3. process_data Function: 
This function now takes a list of DataProcessingStep objects, ensuring that any subclass of DataProcessingStep can be used interchangeably.

By adhering to LSP, you can add new processing steps by creating new subclasses of DataProcessingStep without changing the existing processing logic. This ensures that the program remains correct and flexible, allowing easy extension with new processing steps.

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

class DataProcessingStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class NormalizeFeature(DataProcessingStep):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        std = np.std(df[self.feature_name])
        mean = np.mean(df[self.feature_name])
        df[self.feature_name] = (df[self.feature_name] - mean) / std
        logging.info(f"Feature '{self.feature_name}' normalized")
        return df

class EncodeCategoricalFeature(DataProcessingStep):
    def __init__(self, feature_name: str, encoded_feature_name: str):
        self.feature_name = feature_name
        self.encoded_feature_name = encoded_feature_name
        self.encoder = LabelEncoder()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.encoded_feature_name] = self.encoder.fit_transform(df[self.feature_name])
        logging.info(f"Feature '{self.feature_name}' encoded to '{self.encoded_feature_name}'")
        return df

class FillNaNFeature(DataProcessingStep):
    def __init__(self, feature_name: str, fill_value):
        self.feature_name = feature_name
        self.fill_value = fill_value

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_name] = df[self.feature_name].fillna(self.fill_value)
        logging.info(f"NaN values in feature '{self.feature_name}' filled with '{self.fill_value}'")
        return df

def process_data(path: str, output_path: str, steps: list[DataProcessingStep]):
    # Load data
    data_loader = DataLoader(path)
    df = data_loader.load_data()

    # Apply processing steps
    for step in steps:
        df = step.apply(df)

    # Save processed data
    data_saver = DataSaver(output_path)
    data_saver.save_data(df)

def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    steps = [
        NormalizeFeature("feature_a"),
        EncodeCategoricalFeature("feature_b", "feature_b_encoded"),
        FillNaNFeature("feature_c", -1)
    ]
    process_data(path, output_path, steps)

if __name__ == "__main__":
    main()
