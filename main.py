# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from helper import fill_col

# Treat 'na', 'NA', 'Unknown', and empty strings as missing values
df_raw = pd.read_csv('./datasets/email_phishing_dataset.csv', encoding='ISO-8859-1', na_values=['na',
                                                                                       'NA', 'Unknown', ''])
df_raw.columns = df_raw.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w]', '',
                                                                              regex=True)
df = df_raw.copy()
missing_before = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_before.index, y=missing_before.values, palette='Reds_r')
plt.title("Missing Values Before Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

if 'Sending_Date' in df.columns:
    # Convert to datetime format; invalid formats become NaT
    df['Sending_Date'] = pd.to_datetime(df['Sending_Date'], errors='coerce')
    # Fill missing dates with the most frequent valid date
    mode_date = df['Sending_Date'].mode(dropna=True)
    if not mode_date.empty:
        df['Sending_Date'] = df['Sending_Date'].fillna(mode_date[0])

for col in ['Sending_Time', 'Day']:
    if col in df.columns:
        df[col] = df[col].astype(str).replace(['nan', 'NaT', 'NA', 'na', '', 'Unknown'], pd.NA)
    mode_value = df[col].mode(dropna=True)
    if not mode_value.empty:
        df[col] = df[col].fillna(mode_value[0])
    else:
        df[col] = df[col].fillna('Unknown')

if 'To' in df.columns:
    df.drop(columns=['To'], inplace=True)

    if 'Logo' in df.columns:
        df.drop(columns=['Logo'], inplace=True)

if 'Sender_Name' in df.columns:
    df.drop(columns=['Sender_Name'], inplace=True)

if 'Closing_Remarks' in df.columns:
    fill_col(df, 'Closing_Remarks')

if 'Coined.Word' in df.columns:
    fill_col(df, 'Coined.Word')

if 'Sender_Email' in df.columns:
    fill_col(df, 'Sender_Email')

if 'Sender_Title' in df.columns:
    fill_col(df, 'Sender_Title')

if 'Url_Title' in df.columns:
    fill_col(df, 'Url_Title')

missing_after = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_after.index, y=missing_after.values, palette='Greens')
plt.title("Missing Values After Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

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

df.to_csv('./datasets/Cleaned_PhishingEmailData.csv', index=False)

print("\nPreview of cleaned dataset:")
print(df.head())
