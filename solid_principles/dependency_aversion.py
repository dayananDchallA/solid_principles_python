'''
The Dependency Inversion Principle (DIP) of SOLID principles states that high-level modules should not depend on low-level modules. 
Both should depend on abstractions (e.g., interfaces), and abstractions should not depend on details. 
Details (implementations) should depend on abstractions.

Here's how you can refactor your code to follow the Dependency Inversion Principle:

1. Define abstract interfaces for data loading, saving, and processing.
2. Implement these interfaces with concrete classes.
3. Use dependency injection to provide the required dependencies to the high-level function.


This principle can be applied by:

Defining abstract interfaces that high-level modules and low-level modules depend on.
Using dependency injection to provide the concrete implementations of these interfaces.
Here's how you can refactor your code to follow DIP:

Define interfaces for data loading, data saving, and data processing steps.
Inject dependencies via constructors or function parameters.

Explanation:
-----------
1. Abstract Interfaces: 
Define interfaces (IDataLoader, IDataSaver, IDataProcessingStep) for data loading, data saving, and data processing steps.
2. Concrete Implementations: 
Implement these interfaces in concrete classes (DataLoader, DataSaver, NormalizeFeature, EncodeCategoricalFeature, FillNaNFeature).
3. Dependency Injection: 
In the process_data function, inject dependencies via function parameters. The function operates on abstractions rather than concrete implementations.
4. High-Level Module: 
The process_data function is a high-level module that depends on the abstract interfaces rather than the concrete implementations, adhering to DIP.

This design allows you to easily substitute different implementations of the interfaces without changing the high-level logic, making the system more modular and adaptable to changes.

'''

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)

# Define abstract interfaces
class IDataLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

class IDataSaver(ABC):
    @abstractmethod
    def save_data(self, df: pd.DataFrame):
        pass

class IDataProcessingStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Concrete implementations
class DataLoader(IDataLoader):
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        logging.info(f"Data loaded from {self.path}: {df.head()}")
        return df

class DataSaver(IDataSaver):
    def __init__(self, output_path: str):
        self.output_path = output_path

    def save_data(self, df: pd.DataFrame):
        df.to_parquet(self.output_path)
        logging.info(f"Data saved to {self.output_path}")

class NormalizeFeature(IDataProcessingStep):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        std = np.std(df[self.feature_name])
        mean = np.mean(df[self.feature_name])
        df[self.feature_name] = (df[self.feature_name] - mean) / std
        logging.info(f"Feature '{self.feature_name}' normalized")
        return df

class EncodeCategoricalFeature(IDataProcessingStep):
    def __init__(self, feature_name: str, encoded_feature_name: str):
        self.feature_name = feature_name
        self.encoded_feature_name = encoded_feature_name
        self.encoder = LabelEncoder()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.encoded_feature_name] = self.encoder.fit_transform(df[self.feature_name])
        logging.info(f"Feature '{self.feature_name}' encoded to '{self.encoded_feature_name}'")
        return df

class FillNaNFeature(IDataProcessingStep):
    def __init__(self, feature_name: str, fill_value):
        self.feature_name = feature_name
        self.fill_value = fill_value

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_name] = df[self.feature_name].fillna(self.fill_value)
        logging.info(f"NaN values in feature '{self.feature_name}' filled with '{self.fill_value}'")
        return df

def process_data(loader: IDataLoader, saver: IDataSaver, steps: list[IDataProcessingStep]):
    # Load data
    df = loader.load_data()

    # Apply processing steps
    for step in steps:
        df = step.apply(df)

    # Save processed data
    saver.save_data(df)

def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    
    loader = DataLoader(path)
    saver = DataSaver(output_path)
    steps = [
        NormalizeFeature("feature_a"),
        EncodeCategoricalFeature("feature_b", "feature_b_encoded"),
        FillNaNFeature("feature_c", -1)
    ]
    
    process_data(loader, saver, steps)

if __name__ == "__main__":
    main()
