'''
Single Responsibility Principle (SRP):

Where: The DataProcessor interface and its implementations (Standardizer, Encoder, NanFiller) demonstrate SRP.
How: Each class has a single responsibility related to processing data (e.g., standardization, encoding, NaN filling).
Open/Closed Principle (OCP):

Where: OCP is indirectly applied through the DataProcessor interface and its implementations.
How: If you need to add a new processing step, you can create a new class that implements DataProcessor without modifying existing code.
Liskov Substitution Principle (LSP):

Where: LSP is not explicitly demonstrated in this code.
How: It applies more to class hierarchies and polymorphism, which are not heavily featured in this particular example.
Interface Segregation Principle (ISP):

Where: ISP is not strictly applied in this code.
How: It would be relevant if DataProcessor had methods that were not needed by all its implementations. In this case, all processors use the same method (process_data).
Dependency Inversion Principle (DIP):

Where: DIP is applied in the DataPipeline class.
How: DataPipeline depends on abstractions (DataProcessor), not on concrete implementations. This allows DataPipeline to work with any class that implements DataProcessor (e.g., Standardizer, Encoder, NanFiller), promoting flexibility and easier testing.
Overall, the refactored code improves maintainability and extensibility by adhering to these SOLID principles.

'''
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)


class DataProcessor(ABC):
    @abstractmethod
    def process_data(self, path: str) -> pd.DataFrame:
        pass


class Standardizer(DataProcessor):
    def process_data(self, df: pd.DataFrame) -> pd.Series:
        # Single Responsibility Principle (SRP):
        # This class has a single responsibility related to standardizing data.
        std = np.std(df["feature_a"])
        mean = np.mean(df["feature_a"])
        standardized_feature = (df["feature_a"] - mean) / std
        return standardized_feature


class Encoder(DataProcessor):
    def process_data(self, df: pd.DataFrame) -> pd.Series:
        # Single Responsibility Principle (SRP):
        # This class has a single responsibility related to encoding data.
        encoder = LabelEncoder()
        encoded_feature = encoder.fit_transform(df["feature_b"])
        return pd.Series(encoded_feature, name="feature_b_encoded")


class NanFiller(DataProcessor):
    def process_data(self, df: pd.DataFrame) -> pd.Series:
        # Single Responsibility Principle (SRP):
        # This class has a single responsibility related to filling NaN values.
        filled_feature = df["feature_c"].fillna(-1)
        return filled_feature


class DataPipeline:
    def __init__(self, processors: list[DataProcessor]):
        self.processors = processors

    def process(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        logging.info(f"Data: {df}")
        for processor in self.processors:
            # Dependency Inversion Principle (DIP):
            # DataPipeline depends on abstractions (DataProcessor),
            # not on concrete implementations. This allows DataPipeline
            # to work with any class that implements DataProcessor.
            df = pd.concat([df, processor.process_data(df)], axis=1)
        return df


def main():
    processors = [Standardizer(), Encoder(), NanFiller()]
    pipeline = DataPipeline(processors)
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    processed_df = pipeline.process(path)
    logging.info(f"Processed data: {processed_df}")
    processed_df.to_parquet(output_path)


if __name__ == "__main__":
    main()
