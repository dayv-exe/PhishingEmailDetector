# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d1 = ('./datasets/email_phishing_dataset.csv', './datasets/cleaned_email_phishing_dataset.csv')
d2 = ('./datasets/Nigerian_Fraud.csv', './datasets/cleaned_Nigerian_Fraud.csv')

inputs = [d1, d2]

for input_data, output_data in inputs:
    # Helper function to clean a column
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


    # === Load and Clean Dataset ===
    # Treat 'na', 'NA', 'Unknown', and empty strings as missing values
    df_raw = pd.read_csv(
        input_data,
        encoding='ISO-8859-1',
        na_values=['na', 'NA', 'Unknown', '']
    )

    # Clean column names: strip spaces, replace invalid chars, etc.
    df_raw.columns = (
        df_raw.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )

    df = df_raw.copy()

    # --- Missing Values Before Cleaning ---
    missing_before = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_before.index, y=missing_before.values, hue=missing_before.index, palette='Reds_r',
                legend=False)
    plt.title("Missing Values Before Cleaning")
    plt.ylabel("Count of Missing Entries")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # === Cleaning Operations ===

    # Handle Sending_Date
    if 'Sending_Date' in df.columns:
        df['Sending_Date'] = pd.to_datetime(df['Sending_Date'], errors='coerce')
        mode_date = df['Sending_Date'].mode(dropna=True)
        if not mode_date.empty:
            df['Sending_Date'] = df['Sending_Date'].fillna(mode_date[0])

    # Handle Sending_Time and Day columns
    for col in ['Sending_Time', 'Day']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaT', 'NA', 'na', '', 'Unknown'], pd.NA)
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna('Unknown')

    # Drop unnecessary columns
    for drop_col in ['To', 'Logo', 'Sender_Name']:
        if drop_col in df.columns:
            df.drop(columns=[drop_col], inplace=True)

    # Clean text-based columns using helper
    for text_col in ['Closing_Remarks', 'Coined.Word', 'Sender_Email', 'Sender_Title', 'Url_Title']:
        if text_col in df.columns:
            fill_col(df, text_col)

    # --- Missing Values After Cleaning ---
    missing_after = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_after.index, y=missing_after.values, hue=missing_after.index, palette='Greens', legend=False)
    plt.title("Missing Values After Cleaning")
    plt.ylabel("Count of Missing Entries")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- Comparison Before and After Cleaning ---
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

    # Save cleaned dataset
    df.to_csv(output_data, index=False)

    # Preview cleaned dataset
    print("\nPreview of cleaned dataset:")
    print(df.head())
