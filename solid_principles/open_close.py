'''
To adhere to the Open/Closed Principle (OCP) of the SOLID principles, you should structure your code so that it is open for extension but closed for modification. This means you can add new functionality without changing existing code.

You can achieve this by defining a base class for the data processing steps and then creating subclasses for each specific processing step. Here’s how you can refactor your code:

1. Define a base class for processing steps.
2. Create specific processing step classes that inherit from the base class.
3. Modify the process function to use these processing steps.

'''

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)

# Base class for processing steps
class ProcessingStep(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Normalization processing step
class NormalizeFeature(ProcessingStep):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        std = np.std(df[self.feature_name])
        mean = np.mean(df[self.feature_name])
        df[self.feature_name] = (df[self.feature_name] - mean) / std
        return df

# Encoding categorical feature processing step
class EncodeCategoricalFeature(ProcessingStep):
    def __init__(self, feature_name: str, encoded_feature_name: str):
        self.feature_name = feature_name
        self.encoded_feature_name = encoded_feature_name
        self.encoder = LabelEncoder()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.encoded_feature_name] = self.encoder.fit_transform(df[self.feature_name])
        return df

# Fill NaN processing step
class FillNaNFeature(ProcessingStep):
    def __init__(self, feature_name: str, fill_value):
        self.feature_name = feature_name
        self.fill_value = fill_value

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_name] = df[self.feature_name].fillna(self.fill_value)
        return df

def process(path: str, output_path: str, steps: list[ProcessingStep]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logging.info(f"Data: {df}")

    for step in steps:
        df = step.process(df)

    logging.info(f"Processed data: {df}")
    df.to_parquet(output_path)
    return df

def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    steps = [
        NormalizeFeature("feature_a"),
        EncodeCategoricalFeature("feature_b", "feature_b_encoded"),
        FillNaNFeature("feature_c", -1)
    ]
    process(path, output_path, steps)

if __name__ == "__main__":
    main()
