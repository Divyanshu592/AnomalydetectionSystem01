import pandas as pd

def basic_cleaning(df: pd.DataFrame):
    df = df.dropna()
    return df
