# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from helper import fill_col


def load_data(filepath: str, na_values: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load CSV data with specified missing value indicators.

    Args:
        filepath: Path to the CSV file
        na_values: List of strings to treat as missing values

    Returns:
        DataFrame with cleaned column names
    """
    if na_values is None:
        na_values = ['na', 'NA', 'Unknown', '']

    df = pd.read_csv(filepath, encoding='ISO-8859-1', na_values=na_values)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)

    return df


def plot_missing_values(df: pd.DataFrame, title: str, palette: str = 'Reds_r',
                        exclude_cols: Optional[List[str]] = None) -> None:
    """
    Create a bar plot showing missing values in the dataset.

    Args:
        df: DataFrame to analyze
        title: Title for the plot
        palette: Color palette for the plot
        exclude_cols: Columns to exclude from the analysis
    """
    if exclude_cols is None:
        exclude_cols = ['Email_Content']

    df_analysis = df.drop(columns=exclude_cols, errors='ignore')
    missing_counts = df_analysis.isnull().sum()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, hue=missing_counts.index,
                palette=palette, legend=False)
    plt.title(title)
    plt.ylabel("Count of Missing Entries")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def clean_datetime_column(df: pd.DataFrame, col_name: str = 'Sending_Date') -> pd.DataFrame:
    """
    Clean datetime column by converting to datetime and filling missing values with mode.

    Args:
        df: DataFrame to clean
        col_name: Name of the datetime column

    Returns:
        DataFrame with cleaned datetime column
    """
    if col_name not in df.columns:
        return df

    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    mode_date = df[col_name].mode(dropna=True)

    if not mode_date.empty:
        df[col_name] = df[col_name].fillna(mode_date[0])

    return df


def clean_categorical_columns(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    """
    Clean categorical columns by standardizing missing values and filling with mode.

    Args:
        df: DataFrame to clean
        col_names: List of column names to clean

    Returns:
        DataFrame with cleaned categorical columns
    """
    na_indicators = ['nan', 'NaT', 'NA', 'na', '', 'Unknown']

    for col in col_names:
        if col not in df.columns:
            continue

        df[col] = df[col].astype(str).replace(na_indicators, pd.NA)
        mode_value = df[col].mode(dropna=True)

        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])
        else:
            df[col] = df[col].fillna('Unknown')

    return df


def drop_columns(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame if they exist.

    Args:
        df: DataFrame to modify
        col_names: List of column names to drop

    Returns:
        DataFrame with specified columns removed
    """
    existing_cols = [col for col in col_names if col in df.columns]
    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)

    return df


def fill_columns_with_helper(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    """
    Fill specified columns using the helper fill_col function.

    Args:
        df: DataFrame to modify
        col_names: List of column names to fill

    Returns:
        DataFrame with filled columns
    """
    for col in col_names:
        if col in df.columns:
            fill_col(df, col)

    return df


def plot_missing_comparison(missing_before: pd.Series, missing_after: pd.Series) -> None:
    """
    Create a comparison plot of missing values before and after cleaning.

    Args:
        missing_before: Series of missing value counts before cleaning
        missing_after: Series of missing value counts after cleaning
    """
    missing_comparison = pd.DataFrame({
        'Before Cleaning': missing_before,
        'After Cleaning': missing_after
    })

    missing_comparison.plot(kind='bar', figsize=(10, 5), color=['red', 'green'])
    plt.title("Comparison of Missing Values Before and After Cleaning")
    plt.ylabel("Count of Missing Entries")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_path: Path for the output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    print("\nPreview of cleaned dataset:")
    print(df.head())


def run_cleaning_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Execute the complete data cleaning pipeline.

    Args:
        input_path: Path to the input CSV file
        output_path: Path for the output CSV file

    Returns:
        Cleaned DataFrame
    """
    # Load data
    df_raw = load_data(input_path)
    df = df_raw.copy()

    # Get missing values before cleaning
    missing_before = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()
    plot_missing_values(df, "Missing Values Before Cleaning", palette='Reds_r')

    # Clean datetime column
    df = clean_datetime_column(df, 'Sending_Date')

    # Clean categorical columns
    df = clean_categorical_columns(df, ['Sending_Time', 'Day'])

    # Drop unnecessary columns
    df = drop_columns(df, ['To', 'Logo', 'Sender_Name'])

    # Fill columns using helper function
    df = fill_columns_with_helper(df, ['Closing_Remarks', 'Coined.Word',
                                       'Sender_Email', 'Sender_Title', 'Url_Title'])

    # Get missing values after cleaning
    missing_after = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()
    plot_missing_values(df, "Missing Values After Cleaning", palette='Greens')

    # Plot comparison
    plot_missing_comparison(missing_before, missing_after)

    # Save cleaned data
    save_cleaned_data(df, output_path)

    return df


# Example usage
if __name__ == "__main__":
    cleaned_df = run_cleaning_pipeline(
        input_path='./datasets/email_phishing_dataset.csv',
        output_path='./datasets/Cleaned_PhishingEmailData.csv'
    )