import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

raw_input_dataset = './datasets/dataset1.csv'

# Load the dataset with robust parsing options
try:
    df_raw = pd.read_csv(raw_input_dataset,
                         encoding='ISO-8859-1',
                         na_values=['na', 'NA', 'Unknown', ''],
                         on_bad_lines='skip',  # Skip problematic lines
                         engine='python',  # Use Python engine for better error handling
                         quoting=1,  # QUOTE_ALL
                         escapechar='\\')
    print("✓ Dataset loaded successfully using Python engine")
except Exception as e:
    print(f"Error with Python engine: {e}")
    print("Trying alternative approach...")
    # Alternative: Try with different settings
    try:
        df_raw = pd.read_csv(raw_input_dataset,
                             encoding='utf-8',
                             na_values=['na', 'NA', 'Unknown', ''],
                             on_bad_lines='skip',
                             engine='python',
                             sep=',',
                             quotechar='"',
                             low_memory=False)
        print("✓ Dataset loaded with UTF-8 encoding")
    except Exception as e2:
        print(f"Error with UTF-8: {e2}")
        # Last resort: try ISO-8859-1 with chunk reading
        print("Attempting to load in chunks...")
        chunks = []
        chunk_size = 1000
        for chunk in pd.read_csv(raw_input_dataset,
                                 encoding='ISO-8859-1',
                                 chunksize=chunk_size,
                                 on_bad_lines='skip',
                                 engine='python',
                                 low_memory=False):
            chunks.append(chunk)
        df_raw = pd.concat(chunks, ignore_index=True)
        print(f"✓ Dataset loaded in {len(chunks)} chunks")

# Clean column names
df_raw.columns = (
    df_raw.columns
    .str.strip()
    .str.lower()
    .str.replace(r'[^a-z0-9]+', '_', regex=True)
    .str.strip('_')
)

# Create working copy
df = df_raw.copy()

print(f"Dataset loaded: {len(df)} records")
print(f"Columns: {list(df.columns)}")

# ==================== REMOVE DUPLICATES ====================
initial_count = len(df)

# Check for exact duplicates across all columns
exact_duplicates = df.duplicated().sum()
if exact_duplicates > 0:
    df = df.drop_duplicates()
    print(f"\n✓ Removed {exact_duplicates} exact duplicate records")

# Check for duplicates based on key columns (sender, receiver, date, subject, body)
# This catches emails that are identical in content even if other fields differ
key_columns = ['sender', 'receiver', 'date', 'subject', 'body']
available_keys = [col for col in key_columns if col in df.columns]

if available_keys:
    content_duplicates = df.duplicated(subset=available_keys, keep='first').sum()
    if content_duplicates > 0:
        df = df.drop_duplicates(subset=available_keys, keep='first')
        print(f"✓ Removed {content_duplicates} content-based duplicate records")

# Reset index after dropping duplicates
df = df.reset_index(drop=True)

final_count = len(df)
total_removed = initial_count - final_count

if total_removed > 0:
    print(f"✓ Total duplicates removed: {total_removed}")
    print(f"✓ Records remaining: {final_count}")
else:
    print(f"✓ No duplicates found - all {final_count} records are unique")

# ==================== VISUALIZATION 1: Missing Values Before Cleaning ====================
missing_before = df.isnull().sum()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(missing_before)), missing_before.values,
              color=sns.color_palette("Reds_r", len(missing_before)))
ax.set_xticks(range(len(missing_before)))
ax.set_xticklabels(missing_before.index, rotation=45, ha='right')
ax.set_title("Missing Values Before Cleaning", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Count of Missing Entries", fontsize=11)
ax.set_xlabel("Columns", fontsize=11)

# Add value labels on bars
for i, (idx, val) in enumerate(missing_before.items()):
    if val > 0:
        ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ==================== DATA CLEANING ====================

# DROP rows with missing labels FIRST (most important)
if 'label' in df.columns:
    rows_before = len(df)
    # Convert everything to string, trim spaces
    df['label'] = df['label'].astype(str).str.strip()

    # Replace all known invalid/non-numeric entries with NA
    df['label'] = df['label'].replace(
        ['', ' ', 'None', 'none', 'NONE', 'nan', 'NaN', 'NULL', 'null'],
        pd.NA
    )
    # Now convert valid digit strings to integers where possible
    # Anything else becomes NA
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # Replace values that are NOT 0 or 1 with NA
    df.loc[~df['label'].isin([0, 1]), 'label'] = pd.NA
    df = df.dropna(subset=['label'])
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"\n✓ Dropped {rows_dropped} rows with missing labels")

    # Convert label to integer
    df['label'] = df['label'].astype(int)
    print(f"✓ Label column converted to integer type")

# Clean 'date' column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # Fill missing dates with the most frequent date (mode)
    if df['date'].notna().any():
        mode_date = df['date'].mode()
        if not mode_date.empty:
            df['date'] = df['date'].fillna(mode_date[0])
            print(f"✓ Filled {missing_before.get('date', 0)} missing dates with mode: {mode_date[0]}")
        else:
            # If no mode exists, use the most recent date
            most_recent = df['date'].max()
            df['date'] = df['date'].fillna(most_recent)
            print(f"✓ Filled {missing_before.get('date', 0)} missing dates with most recent: {most_recent}")

# Clean 'sender' column - extract domain only
if 'sender' in df.columns:
    # First, replace any sender without @ with unknown@unknown.unknown
    df['sender'] = df['sender'].fillna("unknown@unknown.unknown")
    df['sender'] = df['sender'].apply(lambda x: x if '@' in str(x) else 'unknown@unknown.unknown')

    # Then extract domain (everything after @)
    df['sender'] = (
        df['sender']
        .str.split('@')
        .str[-1]  # Get everything after @
        .str.extract(r'^([a-zA-Z0-9.]+)', expand=False)  # Extract only letters, numbers, and dots
        .fillna('unknown.unknown')  # Handle any extraction failures
    )

    print("✓ Extracted domain from sender column")

# Drop 'receiver' column
if 'receiver' in df.columns:
    df = df.drop('receiver', axis=1)
    print(f"✓ Dropped 'receiver' column")

# Clean 'urls' column
if 'urls' in df.columns:
    df['urls'] = df['urls'].fillna(0)  # No URLs
    df['urls'] = df['urls'].astype(int)  # Convert to integer
    print(f"✓ Filled {missing_before.get('urls', 0)} missing URL values")

# ==================== VISUALIZATION 2: Missing Values After Cleaning ====================
missing_after = df.isnull().sum()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(missing_after)), missing_after.values,
              color=sns.color_palette("Greens_r", len(missing_after)))
ax.set_xticks(range(len(missing_after)))
ax.set_xticklabels(missing_after.index, rotation=45, ha='right')
ax.set_title("Missing Values After Cleaning", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Count of Missing Entries", fontsize=11)
ax.set_xlabel("Columns", fontsize=11)

# Add value labels on bars
for i, (idx, val) in enumerate(missing_after.items()):
    if val > 0:
        ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ==================== VISUALIZATION 3: Before vs After Comparison ====================
comparison_df = pd.DataFrame({
    'Before Cleaning': missing_before,
    'After Cleaning': missing_after
})

fig, ax = plt.subplots(figsize=(12, 6))
comparison_df.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], alpha=0.8)
ax.set_title("Missing Values: Before vs After Cleaning", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Count of Missing Entries", fontsize=11)
ax.set_xlabel("Columns", fontsize=11)
ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax.legend(title="Status", fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== ADDITIONAL VISUALIZATIONS ====================

# Distribution of labels (Phishing vs Legitimate)
if 'label' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    label_counts = df['label'].value_counts()
    colors = ['#3498db', '#e74c3c']
    ax.pie(label_counts.values, labels=['Legitimate', 'Phishing'], autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 12})
    ax.set_title("Email Label Distribution", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# ==================== SAVE CLEANED DATA ====================
output_file = 'cleaned_datasets/Cleaned_PhishingEmailData.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Cleaned dataset saved to: {output_file}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("DATA CLEANING SUMMARY")
print("=" * 60)
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nMissing values eliminated:")
for col in missing_before.index:
    if col in missing_after.index:
        reduction = missing_before[col] - missing_after[col]
        if reduction > 0:
            print(f"  • {col}: {reduction} values filled/corrected")
    elif col == 'receiver':
        print(f"  • {col}: Column was dropped entirely")
    elif missing_before[col] > 0 and col not in missing_after.index:
        print(f"  • {col}: Column was dropped entirely")

print("\n" + "=" * 60)
print("\nPreview of cleaned dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())