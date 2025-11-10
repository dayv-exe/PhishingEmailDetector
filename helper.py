import pandas as pd

def fill_col(df, col):
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            ['nan', 'na', 'n/a', 'none', 'null', '', 'no data', 'nil'],
            pd.NA
        )
    )

    df[col] = df[col].str.replace(r'[^a-z0-9\s.,!?]', '', regex=True)
    df[col] = df[col].str.capitalize()
    df[col] = df[col].fillna('No data provided')
