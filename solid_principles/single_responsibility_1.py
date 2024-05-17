"""Single responsability:
    * Divide responsability into smaller modules.
    * Only one reason to change
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List


# Good code
def process(path: str, output_path: str) -> None:
    """"""
    df = load_data(path)
    normalized_feature = normalize_feature(df["feature_a"])
    encoded_feature = encode_feature(df["feature_b"])
    filled_feature = fill_feature(df["feature_c"])
    processed_df = compose_df(
        normalized_feature, 
        encoded_feature, 
        filled_feature,
        column_names=df.columns
    )
    save_df(df=processed_df, path=output_path)


def load_data(path: str) -> pd.DataFrame:
    """"""
    return pd.read_parquet(path)
    

def normalize_feature(feature: pd.Series) -> pd.Series:
    """"""
    std = np.std(feature)
    mean = np.mean(feature)
    return (feature - mean) / std


def encode_feature(feature: pd.Series) -> pd.Series:
    """"""
    encoder = LabelEncoder()
    array = encoder.fit_transform(feature)
    return pd.Series(array, name=feature.name)


def fill_feature(feature: pd.Series, value: int = -1) -> pd.Series:
    """"""
    return feature.fillna(value=value)


def compose_df(*args, column_names: List[str]) -> pd.DataFrame:
    """"""
    data = {column_name: series for column_name, series in zip(column_names, args) }
    return pd.DataFrame(data) 


def save_df(df: pd.DataFrame, path: str) -> None:
    """"""
    df.to_parquet(path)


def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    process(path, output_path)


if __name__ == "__main__":
    main()