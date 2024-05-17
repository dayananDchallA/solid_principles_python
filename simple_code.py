import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

def process(path: str, output_path: str) -> pd.DataFrame:
    """"""
    df = pd.read_parquet(path)
    logging.info(f"Data: {df}")
    
    # Normalization
    std = np.std(df["feature_a"])
    mean = np.mean(df["feature_a"])
    standardized_feature = (df["feature_a"] - mean) / std

    # Categorical value
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(df["feature_b"])
    
    # Nan
    filled_feature = df["feature_c"].fillna(-1)

    # Convert encoded feature to a pandas Series
    encoded_feature = pd.Series(encoded_feature, name="feature_b_encoded")

    processed_df = pd.concat(
        [standardized_feature, encoded_feature, filled_feature],
        axis=1
    )
    logging.info(f"Processed data: {processed_df}")
    processed_df.to_parquet(output_path)


def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    process(path, output_path)


if __name__ == "__main__":
    main()