# sentence count
# ave sentence length
# unique_word_ratio = unique_words / body_word_count
# phone number count
# has phone number
# special char count
# email tone
# convert body to lower case

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from spellchecker import SpellChecker

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load the cleaned dataset
try:
    df = pd.read_csv('cleaned_datasets/Cleaned_PhishingEmailData.csv')
    print("✓ Cleaned dataset loaded successfully")
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print("Error: Cleaned_PhishingEmailData.csv not found!")
    print("Please run datacleaning.py first to generate the cleaned dataset.")
    exit(1)

# ==================== FEATURE ENGINEERING ====================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ==================== DATE-TIME FEATURES ====================
if 'date' in df.columns:
    # Convert date column to datetime if it isn't already
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    print("\n✓ Extracting date-time features...")

    # Extract day of the week (Monday=0, Sunday=6)
    df['day_of_week'] = df['date'].dt.dayofweek

    # Extract day of the week name for better readability
    df['day_name'] = df['date'].dt.day_name().str.lower()

    # Extract month (1-12)
    df['month'] = df['date'].dt.month

    # Extract month name for better readability
    df['month_name'] = df['date'].dt.month_name().str.lower()

    # Extract time (HH:MM:SS format)
    df['time'] = df['date'].dt.time

    # Extract year for visualization
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        print(f"  • Created 'year' column for visualization")

    # Extract hour for potential analysis
    df['hour'] = df['date'].dt.hour

    print(f"  • Created 'day_of_week' column (0=Monday, 6=Sunday)")
    print(f"  • Created 'day_name' column (e.g., Monday, Tuesday)")
    print(f"  • Created 'month' column (1-12)")
    print(f"  • Created 'month_name' column (e.g., January, February)")
    print(f"  • Created 'time' column (HH:MM:SS format)")
    print(f"  • Created 'hour' column (0-23)")

    # Drop the original date column after extraction
    df = df.drop('date', axis=1)
    print(f"✓ Dropped 'date' column after extracting all components")

# ==================== DOMAIN FEATURES ====================
if 'sender' in df.columns:
    print("\n✓ Extracting domain features...")

    # Extract top-level domain (TLD) and second-level domain (SLD)
    domain_parts = df['sender'].str.split('.')

    # Top-level domain is the last part (e.g., 'com', 'org', 'uk')
    df['top_level_domain'] = domain_parts.str[-1].str.lower()

    # Second-level domain is the second to last part (e.g., 'example' in 'example.com')
    df['second_level_domain'] = domain_parts.str[-2].fillna('unknown').str.lower()

    print("  • Created 'top_level_domain' column (e.g., com, org, net)")
    print("  • Created 'second_level_domain' column (e.g., google, amazon)")

# ==================== DOMAIN ENCODING ====================
print("\n✓ Encoding domain features...")

# Convert to lowercase and encode top-level domain
if 'top_level_domain' in df.columns:
    df['top_level_domain'] = df['top_level_domain'].str.lower()
    df['top_level_domain_encoded'], tld_uniques = pd.factorize(df['top_level_domain'])
    print(f"  • Converted 'top_level_domain' to lowercase")
    print(f"  • Created 'top_level_domain_encoded' column")
    print(f"    - Unique TLDs: {len(tld_uniques)}")
    print(f"    - Encoded range: 0 to {df['top_level_domain_encoded'].max()}")

# Convert to lowercase and encode second-level domain
if 'second_level_domain' in df.columns:
    df['second_level_domain'] = df['second_level_domain'].str.lower()
    df['second_level_domain_encoded'], sld_uniques = pd.factorize(df['second_level_domain'])
    print(f"  • Converted 'second_level_domain' to lowercase")
    print(f"  • Created 'second_level_domain_encoded' column")
    print(f"    - Unique SLDs: {len(sld_uniques)}")
    print(f"    - Encoded range: 0 to {df['second_level_domain_encoded'].max()}")

# ==================== URL/LINK COUNTING ====================
print("\n✓ Counting URLs/links in text...")

import re

def count_urls(text):
    """Count the number of URLs in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    # Pattern to match URLs (http://, https://, www., or common TLDs)
    url_pattern = r'(?:http[s]?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return len(urls)

# Update the 'urls' column with total count from subject and body
if 'urls' in df.columns:
    subject_count = df['subject'].apply(count_urls) if 'subject' in df.columns else 0
    body_count = df['body'].apply(count_urls) if 'body' in df.columns else 0
    df['urls'] = subject_count + body_count
    print(f"  • Updated 'urls' column with link counts from subject and body")
    print(f"    - Total URLs across all emails: {df['urls'].sum()}")
    print(f"    - Average URLs per email: {df['urls'].mean():.2f}")
    print(f"    - Emails with 0 URLs: {(df['urls'] == 0).sum()}")
    print(f"    - Emails with 1+ URLs: {(df['urls'] > 0).sum()}")
    # Create binary has_url column
    df['has_url'] = (df['urls'] > 0).astype(int)
    print(f"  • Created 'has_url' column (1 if URL present, 0 otherwise)")
    print(f"    - Emails with URLs (has_url=1): {df['has_url'].sum()}")
    print(f"    - Emails without URLs (has_url=0): {(df['has_url'] == 0).sum()}")

# ==================== PHONE NUMBER FEATURES ====================
def count_phone_numbers(text):
    """Count the number of phone numbers in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0

    # Pattern to match various phone number formats:
    # (123) 456-7890, 123-456-7890, 123.456.7890, 1234567890, +1 123 456 7890, etc.
    phone_pattern = r'(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)'
    phones = re.findall(phone_pattern, text)
    return len(phones)


# Create phone number count features for subject and body
if 'subject' in df.columns and 'body' in df.columns:
    subject_phone_count = df['subject'].apply(count_phone_numbers)
    body_phone_count = df['body'].apply(count_phone_numbers)
    df['phone_number_count'] = subject_phone_count + body_phone_count

    print("  • Created 'phone_numbers' column")
    print(f"    - Total phone numbers across all emails: {df['phone_number_count'].sum()}")
    print(f"    - Average phone numbers per email: {df['phone_number_count'].mean():.2f}")
    print(f"    - Emails with 0 phone numbers: {(df['phone_number_count'] == 0).sum()}")
    print(f"    - Emails with 1+ phone numbers: {(df['phone_number_count'] > 0).sum()}")

    # Create binary has_phone_number column
    df['has_phone_number'] = (df['phone_number_count'] > 0).astype(int)
    print("  • Created 'has_phone_number' column (1 if phone number present, 0 otherwise)")
    print(f"    - Emails with phone numbers (has_phone_number=1): {df['has_phone_number'].sum()}")
    print(f"    - Emails without phone numbers (has_phone_number=0): {(df['has_phone_number'] == 0).sum()}")

# ==================== TYPO DETECTION ====================
print("\n✓ Detecting typos...")

spell = SpellChecker()

def count_typos(text):
    """Count the number of misspelled words in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0

    # Split text into words and convert to lowercase
    words = text.lower().split()

    # Filter out words that are purely numeric or contain numbers
    words = [word.strip('.,!?;:()[]{}"\'-') for word in words]
    words = [word for word in words if word and not any(char.isdigit() for char in word)]

    if len(words) == 0:
        return 0

    # Find misspelled words
    misspelled = spell.unknown(words)
    return len(misspelled)


def typo_percentage(text):
    """Calculate percentage of misspelled words in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0.0

    # Split text into words
    words = text.lower().split()

    # Filter out words that are purely numeric or contain numbers
    words = [word.strip('.,!?;:()[]{}"\'-') for word in words]
    words = [word for word in words if word and not any(char.isdigit() for char in word)]

    if len(words) == 0:
        return 0.0

    # Find misspelled words
    misspelled = spell.unknown(words)
    return len(misspelled) / len(words)


# Count typos in subject and body
if 'subject' in df.columns:
    print("  • Analyzing subject typos (this may take a moment)...")
    df['subject_typo_count'] = df['subject'].apply(count_typos)
    df['subject_typo_pct'] = df['subject'].apply(typo_percentage)
    print("  • Created 'subject_typo_count' column")
    print(f"    - Average typos in subject: {df['subject_typo_count'].mean():.2f}")
    print("  • Created 'subject_typo_pct' column (0-1 scale)")
    print(f"    - Average typo percentage: {df['subject_typo_pct'].mean():.4f}")

if 'body' in df.columns:
    print("  • Analyzing body typos (this may take a while)...")
    df['body_typo_count'] = df['body'].apply(count_typos)
    df['body_typo_pct'] = df['body'].apply(typo_percentage)
    print("  • Created 'body_typo_count' column")
    print(f"    - Average typos in body: {df['body_typo_count'].mean():.2f}")
    print("  • Created 'body_typo_pct' column (0-1 scale)")
    print(f"    - Average typo percentage: {df['body_typo_pct'].mean():.4f}")

# Combined typo metrics
if 'subject' in df.columns and 'body' in df.columns:
    df['total_typo_count'] = df['subject_typo_count'] + df['body_typo_count']
    print("  • Created 'total_typo_count' column")
    print(f"    - Total typos across all emails: {df['total_typo_count'].sum()}")
    print(f"    - Average total typos per email: {df['total_typo_count'].mean():.2f}")

# ==================== SPECIAL CHARACTER COUNTING ====================
print("\n✓ Counting special characters...")

def count_special_chars(text):
    """Count special characters (non-alphanumeric, non-whitespace) in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    # Count characters that are not letters, digits, or whitespace
    special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace() and char not in '.?,')
    return special_char_count

def special_char_percentage(text):
    """Calculate percentage of special characters in text"""
    if not isinstance(text, str) or pd.isna(text) or len(text) == 0:
        return 0.0
    special_count = sum(1 for char in text if not char.isalnum() and not char.isspace() and char not in '.?,')
    return special_count / len(text)

# Count special characters in subject and body
if 'subject' in df.columns:
    df['subject_special_char_count'] = df['subject'].apply(count_special_chars)
    df['subject_special_char_pct'] = df['subject'].apply(special_char_percentage)
    print("  • Created 'subject_special_char_count' column")
    print(f"    - Average special chars in subject: {df['subject_special_char_count'].mean():.2f}")
    print("  • Created 'subject_special_char_pct' column (0-1 scale)")
    print(f"    - Average special char percentage: {df['subject_special_char_pct'].mean():.4f}")

if 'body' in df.columns:
    df['body_special_char_count'] = df['body'].apply(count_special_chars)
    df['body_special_char_pct'] = df['body'].apply(special_char_percentage)
    print("  • Created 'body_special_char_count' column")
    print(f"    - Average special chars in body: {df['body_special_char_count'].mean():.2f}")
    print("  • Created 'body_special_char_pct' column (0-1 scale)")
    print(f"    - Average special char percentage: {df['body_special_char_pct'].mean():.4f}")

# Combined special character count
if 'subject' in df.columns and 'body' in df.columns:
    df['total_special_char_count'] = df['subject_special_char_count'] + df['body_special_char_count']
    print("  • Created 'total_special_char_count' column")
    print(f"    - Total special chars across all emails: {df['total_special_char_count'].sum()}")
    print(f"    - Average total special chars per email: {df['total_special_char_count'].mean():.2f}")

# ==================== NON-ASCII CHARACTERS COUNTING ====================
print("\n✓ Counting non-ASCII characters...")


def count_non_ascii_chars(text):
    """Count non-ASCII characters in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    return non_ascii_count


def non_ascii_percentage(text):
    """Calculate percentage of non-ASCII characters in text"""
    if not isinstance(text, str) or pd.isna(text) or len(text) == 0:
        return 0.0
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    return non_ascii_count / len(text)


# Calculate non-ASCII percentage in subject and body combined
if 'subject' in df.columns and 'body' in df.columns:
    # Combine subject and body text
    combined_text = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df['non_ascii_char_pct'] = combined_text.apply(non_ascii_percentage)

    print(f"  • Created 'non_ascii_char_pct' column (0-1 scale)")
    print(f"    - Average non-ASCII percentage: {df['non_ascii_char_pct'].mean():.4f}")
    print(f"    - Emails with 0% non-ASCII: {(df['non_ascii_char_pct'] == 0).sum()}")
    print(f"    - Emails with >1% non-ASCII: {(df['non_ascii_char_pct'] > 0.01).sum()}")
    print(f"    - Emails with >5% non-ASCII: {(df['non_ascii_char_pct'] > 0.05).sum()}")
    print(f"    - Max non-ASCII percentage: {df['non_ascii_char_pct'].max():.4f}")

    # Create binary feature for presence of non-ASCII characters
    df['has_non_ascii_chars'] = (df['non_ascii_char_pct'] > 0).astype(int)
    print(f"  • Created 'has_non_ascii_chars' column (1 if present, 0 otherwise)")

# ==================== SENTENCE COUNT FEATURE ====================
import re


def count_sentences(text):
    """Count the number of sentences in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0

    # Pattern to match sentence-ending punctuation (., !, ?)
    # followed by whitespace or end of string
    sentence_pattern = r'[.!?]+(?:\s|$)'
    sentences = re.findall(sentence_pattern, text)

    # If no sentence-ending punctuation found but text exists, count as 1 sentence
    if len(sentences) == 0 and len(text.strip()) > 0:
        return 1

    return len(sentences)


def average_sentence_length(text):
    """Calculate average sentence length in words, removing outliers"""
    if not isinstance(text, str) or pd.isna(text):
        return 0

    # Split text into sentences using sentence-ending punctuation
    sentence_pattern = r'[.!?]+'
    sentences = re.split(sentence_pattern, text)

    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        return 0

    # Count words in each sentence
    sentence_lengths = [len(sentence.split()) for sentence in sentences]

    # Remove outliers using IQR method if we have enough sentences
    if len(sentence_lengths) > 4:
        q1 = pd.Series(sentence_lengths).quantile(0.25)
        q3 = pd.Series(sentence_lengths).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out outliers
        sentence_lengths = [sl for sl in sentence_lengths if lower_bound <= sl <= upper_bound]

    # Return average of non-outlier sentence lengths
    if len(sentence_lengths) == 0:
        return 0
    return sum(sentence_lengths) / len(sentence_lengths)


# Create sentence count and average sentence length features for body
if 'body' in df.columns:
    df['body_sentence_count'] = df['body'].apply(count_sentences)
    print("  • Created 'body_sentence_count' column")
    print(f"    - Average sentences per email: {df['body_sentence_count'].mean():.2f}")
    print(f"    - Min sentences: {df['body_sentence_count'].min()}")
    print(f"    - Max sentences: {df['body_sentence_count'].max()}")
    print(f"    - Emails with 0 sentences: {(df['body_sentence_count'] == 0).sum()}")

    df['body_avg_sentence_length'] = df['body'].apply(average_sentence_length)
    print("  • Created 'body_avg_sentence_length' column")
    print(f"    - Average sentence length: {df['body_avg_sentence_length'].mean():.2f} words")
    print(f"    - Min avg sentence length: {df['body_avg_sentence_length'].min():.2f}")
    print(f"    - Max avg sentence length: {df['body_avg_sentence_length'].max():.2f}")

# ==================== TEXT LENGTH FEATURES ====================
print("\n✓ Creating text length features...")

def word_count(text):
    """Count the number of words in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    return len(str(text).split())


def average_word_length(text):
    """Calculate average word length in text, removing outliers"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    words = str(text).split()
    if len(words) == 0:
        return 0

    # Get word lengths
    word_lengths = [len(word) for word in words]

    # Remove outliers using IQR method if we have enough words
    if len(word_lengths) > 4:
        q1 = pd.Series(word_lengths).quantile(0.25)
        q3 = pd.Series(word_lengths).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out outliers
        word_lengths = [wl for wl in word_lengths if lower_bound <= wl <= upper_bound]

    # Return average of non-outlier word lengths
    if len(word_lengths) == 0:
        return 0
    return sum(word_lengths) / len(word_lengths)

# Create word count features
if 'subject' in df.columns:
    df['subject_word_count'] = df['subject'].apply(word_count)
    print("  • Created 'subject_word_count' column")

if 'body' in df.columns:
    df['body_word_count'] = df['body'].apply(word_count)
    print("  • Created 'body_word_count' column")

# Create average word length features
if 'subject' in df.columns:
    df['subject_avg_word_length'] = df['subject'].apply(average_word_length)
    print("  • Created 'subject_avg_word_length' column")

if 'body' in df.columns:
    df['body_avg_word_length'] = df['body'].apply(average_word_length)
    print("  • Created 'body_avg_word_length' column")

# ==================== UNIQUE WORDS FEATURES ====================

def count_unique_words(text):
    """Count the number of unique words in text"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    words = str(text).lower().split()  # Convert to lowercase for accurate uniqueness
    return len(set(words))


# Create unique word ratio feature for body
if 'body' in df.columns and 'body_word_count' in df.columns:
    unique_word_count = df['body'].apply(count_unique_words)
    print(f"    - Average unique words per email: {unique_word_count.mean():.2f}")

    # Calculate unique word ratio (avoid division by zero) - vectorized operation
    df['body_unique_word_ratio'] = unique_word_count / df['body_word_count'].replace(0, 1)
    df.loc[df['body_word_count'] == 0, 'body_unique_word_ratio'] = 0

    print("  • Created 'body_unique_word_ratio' column")
    print(f"    - Average unique word ratio: {df['body_unique_word_ratio'].mean():.4f}")
    print(f"    - Min ratio: {df['body_unique_word_ratio'].min():.4f}")
    print(f"    - Max ratio: {df['body_unique_word_ratio'].max():.4f}")
    print(f"    - Emails with ratio = 1.0 (all unique): {(df['body_unique_word_ratio'] == 1.0).sum()}")

# ==================== SUSPICIOUS/SPAMMY WORDS DETECTION ====================
print("\n✓ Detecting suspicious/spammy words and phrases...")

# Load suspicious words/phrases from sus.txt
try:
    with open('sus.txt', 'r', encoding='utf-8') as f:
        # Read all lines and strip whitespace
        suspicious_terms = [line.strip().lower() for line in f if line.strip()]

    # Separate single words from multi-word phrases
    suspicious_words = [term for term in suspicious_terms if ' ' not in term]
    suspicious_phrases = [term for term in suspicious_terms if ' ' in term]

    print(f"  • Loaded {len(suspicious_terms)} suspicious terms from sus.txt")
    print(f"    - Single words: {len(suspicious_words)}")
    print(f"    - Multi-word phrases: {len(suspicious_phrases)}")


    def count_suspicious_terms(text):
        """Count the number of suspicious words and phrases in text"""
        if not isinstance(text, str) or pd.isna(text):
            return 0

        text_lower = text.lower()
        count = 0

        # Count multi-word phrases and add the word count for each match
        for phrase in suspicious_phrases:
            matches = text_lower.count(phrase)
            if matches > 0:
                # Count each word in the phrase for each match
                word_count = len(phrase.split())
                count += matches * word_count

        # Count single words
        words = text_lower.split()
        for word in words:
            # Remove punctuation from word
            clean_word = word.strip('.,!?;:()[]{}"\'-')
            if clean_word in suspicious_words:
                count += 1

        return count

    def get_suspicious_terms_found(text):
        """Get a list of all suspicious terms found in text"""
        if not isinstance(text, str) or pd.isna(text):
            return []

        text_lower = text.lower()
        found_terms = []

        # Find multi-word phrases
        for phrase in suspicious_phrases:
            if phrase in text_lower:
                found_terms.append(phrase)

        # Find single words
        words = text_lower.split()
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}"\'-')
            if clean_word in suspicious_words:
                found_terms.append(clean_word)

        return found_terms


    def suspicious_term_percentage(text):
        """Calculate percentage of suspicious terms relative to total words"""
        if not isinstance(text, str) or pd.isna(text):
            return 0.0

        words = text.split()
        if len(words) == 0:
            return 0.0

        suspicious_count = count_suspicious_terms(text)
        return suspicious_count / len(words)


    # Create suspicious term features for subject
    if 'subject' in df.columns:
        df['subject_suspicious_count'] = df['subject'].apply(count_suspicious_terms)
        df['subject_suspicious_pct'] = df['subject'].apply(suspicious_term_percentage)
        print("  • Created 'subject_suspicious_count' column")
        print(f"    - Average suspicious terms in subject: {df['subject_suspicious_count'].mean():.2f}")
        print(f"    - Emails with suspicious terms in subject: {(df['subject_suspicious_count'] > 0).sum()}")
        print("  • Created 'subject_suspicious_pct' column (0–1)")
        print(f"    - Average suspicious term percentage: {df['subject_suspicious_pct'].mean():.4f}")
        print("  • Created 'subject_suspicious_terms_found' column (list of found terms)")

    # Create suspicious term features for body
    if 'body' in df.columns:
        df['body_suspicious_count'] = df['body'].apply(count_suspicious_terms)
        df['body_suspicious_pct'] = df['body'].apply(suspicious_term_percentage)
        print("  • Created 'body_suspicious_count' column")
        print(f"    - Average suspicious terms in body: {df['body_suspicious_count'].mean():.2f}")
        print(f"    - Emails with suspicious terms in body: {(df['body_suspicious_count'] > 0).sum()}")
        print("  • Created 'body_suspicious_pct' column (0–1)")
        print(f"    - Average suspicious term percentage: {df['body_suspicious_pct'].mean():.4f}")
        print("  • Created 'body_suspicious_terms_found' column (list of found terms)")

except FileNotFoundError:
    print("  ⚠ Warning: sus.txt not found! Skipping suspicious word detection.")
    print("    Create a sus.txt file with suspicious words/phrases (one per line) to enable this feature.")
except Exception as e:
    print(f"  ⚠ Error loading sus.txt: {e}")

# ==================== TEXT UPPERCASE PERCENTAGE FEATURES ====================
print("\n✓ Creating uppercase percentage features...")

def uppercase_percentage(text):
    if not isinstance(text, str):
        return 0
    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    return upper / alpha if alpha > 0 else 0  # Avoid division by zero

def count_uppercase_words(text):
    """Count the number of words that are entirely uppercase"""
    if not isinstance(text, str) or pd.isna(text):
        return 0
    words = text.split()
    # Count words that are all uppercase and contain at least one letter
    uppercase_words = sum(1 for word in words if word.isupper() and any(c.isalpha() for c in word))
    return uppercase_words

def uppercase_word_percentage(text):
    """Calculate percentage of words that are entirely uppercase"""
    if not isinstance(text, str) or pd.isna(text):
        return 0.0
    words = text.split()
    # Filter to words that contain at least one letter
    words_with_letters = [word for word in words if any(c.isalpha() for c in word)]
    if len(words_with_letters) == 0:
        return 0.0
    uppercase_words = sum(1 for word in words_with_letters if word.isupper())
    return uppercase_words / len(words_with_letters)

# Create uppercase character percentage for subject and body
if 'subject' in df.columns:
    df['subject_uppercase_pct'] = df['subject'].apply(uppercase_percentage)
    print("  • Created 'subject_uppercase_pct' column (0–1)")

if 'body' in df.columns:
    df['body_uppercase_pct'] = df['body'].apply(uppercase_percentage)
    print("  • Created 'body_uppercase_pct' column (0–1)")

# Create uppercase word count and percentage for subject and body
if 'subject' in df.columns:
    df['subject_uppercase_word_count'] = df['subject'].apply(count_uppercase_words)
    df['subject_uppercase_word_pct'] = df['subject'].apply(uppercase_word_percentage)
    print("  • Created 'subject_uppercase_word_count' column")
    print(f"    - Average uppercase words in subject: {df['subject_uppercase_word_count'].mean():.2f}")
    print("  • Created 'subject_uppercase_word_pct' column (0–1)")
    print(f"    - Average uppercase word percentage: {df['subject_uppercase_word_pct'].mean():.4f}")

if 'body' in df.columns:
    df['body_uppercase_word_count'] = df['body'].apply(count_uppercase_words)
    df['body_uppercase_word_pct'] = df['body'].apply(uppercase_word_percentage)
    print("  • Created 'body_uppercase_word_count' column")
    print(f"    - Average uppercase words in body: {df['body_uppercase_word_count'].mean():.2f}")
    print("  • Created 'body_uppercase_word_pct' column (0–1)")
    print(f"    - Average uppercase word percentage: {df['body_uppercase_word_pct'].mean():.4f}")

# ==================== VISUALIZATIONS ====================
print("\n✓ Creating visualizations...")

def remove_outliers_iqr(data):
    """Remove outliers using IQR method"""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Visualization: Emails by Year
if 'year' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    year_counts = df['year'].value_counts().sort_index()
    ax.bar(year_counts.index, year_counts.values, color='#e74c3c', alpha=0.8)
    ax.set_title("Email Distribution by Year", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    # Zoom in to show only the range of years with data
    ax.set_xlim(year_counts.index.min() - 0.5, year_counts.index.max() + 0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualization: Emails by Day of Week
if 'day_name' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    day_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    day_counts = df['day_name'].value_counts().reindex(day_order, fill_value=0)
    ax.bar(range(len(day_counts)), day_counts.values, color='#3498db', alpha=0.8)
    ax.set_title("Email Distribution by Day of Week", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_xticks(range(len(day_counts)))
    ax.set_xticklabels(day_counts.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Visualization: Emails by Month
if 'month_name' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    month_order = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']
    month_counts = df['month_name'].value_counts().reindex(month_order, fill_value=0)
    ax.bar(range(len(month_counts)), month_counts.values, color='#2ecc71', alpha=0.8)
    ax.set_title("Email Distribution by Month", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_xticks(range(len(month_counts)))
    ax.set_xticklabels(month_counts.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Visualization: Emails by Hour of Day
if 'hour' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    hour_counts = df['hour'].value_counts().sort_index()
    ax.bar(hour_counts.index, hour_counts.values, color='#9b59b6', alpha=0.8)
    ax.set_title("Email Distribution by Hour of Day", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Hour (0-23)", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualization: Top-Level Domains Distribution
if 'top_level_domain' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    tld_counts = df['top_level_domain'].value_counts().head(15)
    ax.bar(range(len(tld_counts)), tld_counts.values, color='#e67e22', alpha=0.8)
    ax.set_title("Top 15 Most Common Top-Level Domains", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Top-Level Domain", fontsize=11)
    ax.set_xticks(range(len(tld_counts)))
    ax.set_xticklabels(tld_counts.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Visualization: Top 15 Second-Level Domains Distribution
if 'second_level_domain' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    sld_counts = df['second_level_domain'].value_counts().head(15)
    ax.bar(range(len(sld_counts)), sld_counts.values, color='#1abc9c', alpha=0.8)
    ax.set_title("Top 15 Most Common Second-Level Domains", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_xlabel("Second-Level Domain", fontsize=11)
    ax.set_xticks(range(len(sld_counts)))
    ax.set_xticklabels(sld_counts.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Visualization: Emails with URLs vs Without URLs (Pie Chart)
if 'has_url' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 8))
    url_counts = df['has_url'].value_counts()
    labels = ['Without URLs', 'With URLs']
    colors = ['#95a5a6', '#3498db']
    explode = (0.05, 0)
    ax.pie(url_counts.values, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors, explode=explode, shadow=True)
    ax.set_title("Distribution of Emails: With vs Without URLs", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# Visualization: Average URL Count in Emails with URLs (with outlier removal)
if 'urls' in df.columns and 'has_url' in df.columns:
    emails_with_urls = df[df['has_url'] == 1]['urls']
    if len(emails_with_urls) > 0:
        # Remove outliers
        filtered_urls = remove_outliers_iqr(emails_with_urls)
        max_urls = emails_with_urls.max()  # Get max from original data

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(filtered_urls, bins=min(50, int(filtered_urls.max())),
                color='#27ae60', alpha=0.8, edgecolor='black')
        ax.axvline(filtered_urls.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {filtered_urls.mean():.2f} URLs')
        ax.axvline(filtered_urls.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {filtered_urls.median():.2f} URLs')
        modes = filtered_urls.mode()
        if len(modes) > 0:
            mode_val = modes.iloc[0]
            ax.axvline(mode_val, color='blue', linestyle='--', linewidth=2,
                       label=f'Mode: {mode_val} URLs')
        ax.set_title(f"Distribution of URL Count in Emails with URLs (Outliers Removed)\nMax Value: {max_urls}",
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlabel("Number of URLs", fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Visualization: Emails with Phone Numbers vs Without Phone Numbers (Pie Chart)
if 'has_phone_number' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 8))
    phone_counts = df['has_phone_number'].value_counts()
    labels = ['Without Phone Numbers', 'With Phone Numbers']
    colors = ['#95a5a6', '#e74c3c']
    explode = (0.05, 0)
    ax.pie(phone_counts.values, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors, explode=explode, shadow=True)
    ax.set_title("Distribution of Emails: With vs Without Phone Numbers",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# Visualization: Average Phone Number Count in Emails with Phone Numbers (with outlier removal)
if 'phone_number_count' in df.columns and 'has_phone_number' in df.columns:
    emails_with_phones = df[df['has_phone_number'] == 1]['phone_number_count']
    if len(emails_with_phones) > 0:
        # Remove outliers
        filtered_phones = remove_outliers_iqr(emails_with_phones)
        max_phones = emails_with_phones.max()  # Get max from original data

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(filtered_phones, bins=min(50, int(filtered_phones.max())),
                color='#f39c12', alpha=0.8, edgecolor='black')
        ax.axvline(filtered_phones.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {filtered_phones.mean():.2f} phone numbers')
        ax.axvline(filtered_phones.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {filtered_phones.median():.2f} phone numbers')
        modes = filtered_phones.mode()
        if len(modes) > 0:
            mode_val = modes.iloc[0]
            ax.axvline(mode_val, color='blue', linestyle='--', linewidth=2,
                       label=f'Mode: {mode_val} phone numbers')
        ax.set_title(f"Distribution of Phone Number Count (Outliers Removed)\nMax Value: {max_phones}",
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlabel("Number of Phone Numbers", fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Visualization: Average Body Sentence Length Distribution (with outlier removal)
if 'body_avg_sentence_length' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    # Filter out zeros for better visualization
    filtered_data = df[df['body_avg_sentence_length'] > 0]['body_avg_sentence_length']
    max_sentence_length = filtered_data.max()  # Get max from original data

    # Remove outliers
    filtered_data_no_outliers = remove_outliers_iqr(filtered_data)

    ax.hist(filtered_data_no_outliers, bins=50, color='#16a085', alpha=0.8, edgecolor='black')
    ax.axvline(filtered_data_no_outliers.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {filtered_data_no_outliers.mean():.2f} words')
    ax.axvline(filtered_data_no_outliers.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {filtered_data_no_outliers.median():.2f} words')
    modes = filtered_data_no_outliers.mode()
    if len(modes) > 0:
        mode_val = modes.iloc[0]
        ax.axvline(mode_val, color='blue', linestyle='--', linewidth=2,
                   label=f'Mode: {mode_val} words')
    ax.set_title(
        f"Distribution of Average Body Sentence Length (Outliers Removed)\nMax Value: {max_sentence_length:.2f} words",
        fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlabel("Average Sentence Length (words)", fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualization: Average Body Word Length Distribution (with outlier removal)
if 'body_avg_word_length' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    # Filter out zeros for better visualization
    filtered_data = df[df['body_avg_word_length'] > 0]['body_avg_word_length']
    max_word_length = filtered_data.max()  # Get max from original data

    # Remove outliers
    filtered_data_no_outliers = remove_outliers_iqr(filtered_data)

    ax.hist(filtered_data_no_outliers, bins=50, color='#8e44ad', alpha=0.8, edgecolor='black')
    ax.axvline(filtered_data_no_outliers.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {filtered_data_no_outliers.mean():.2f} characters')
    ax.axvline(filtered_data_no_outliers.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {filtered_data_no_outliers.median():.2f} characters')
    modes = filtered_data_no_outliers.mode()
    if len(modes) > 0:
        mode_val = modes.iloc[0]
        ax.axvline(mode_val, color='green', linestyle='--', linewidth=2,
                   label=f'Mode: {mode_val} characters')
    ax.set_title(
        f"Distribution of Average Body Word Length (Outliers Removed)\nMax Value: {max_word_length:.2f} characters",
        fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlabel("Average Word Length (characters)", fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualization: Body Unique Word Ratio Distribution
if 'body_unique_word_ratio' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    # Filter out zeros for better visualization
    filtered_data = df[df['body_unique_word_ratio'] > 0]['body_unique_word_ratio']
    ax.hist(filtered_data, bins=50, color='#d35400', alpha=0.8, edgecolor='black')
    ax.axvline(filtered_data.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {filtered_data.mean():.4f}')
    ax.axvline(filtered_data.median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {filtered_data.median():.4f}')
    modes = filtered_data.mode()
    if len(modes) > 0:
        mode_val = modes.iloc[0]
        ax.axvline(mode_val, color='yellow', linestyle='--', linewidth=2,
                   label=f'Mode: {mode_val}')
    ax.set_title("Distribution of Body Unique Word Ratio", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlabel("Unique Word Ratio (0-1)", fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualization: Body Uppercase Percentage Distribution
if 'body_uppercase_pct' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['body_uppercase_pct'], bins=50, color='#c0392b', alpha=0.8, edgecolor='black')
    ax.axvline(df['body_uppercase_pct'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["body_uppercase_pct"].mean():.4f}')
    ax.axvline(df['body_uppercase_pct'].median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {df["body_uppercase_pct"].median():.4f}')
    modes = df['body_uppercase_pct'].mode()
    if len(modes) > 0:
        mode_val = modes.iloc[0]
        ax.axvline(mode_val, color='green', linestyle='--', linewidth=2,
                   label=f'Mode: {mode_val}')
    ax.set_title("Distribution of Body Uppercase Percentage", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlabel("Uppercase Percentage (0-1)", fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== SAVE FEATURE ENGINEERED DATA ====================
output_file = 'cleaned_datasets/FeatureEngineered_PhishingEmailData.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Feature-engineered dataset saved to: {output_file}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print(f"\nNew features created:")
if 'day_of_week' in df.columns:
    print(f"  • day_of_week: Numeric day (0=Monday, 6=Sunday)")
if 'day_name' in df.columns:
    print(f"  • day_name: Day name (e.g., Monday, Tuesday)")
if 'month' in df.columns:
    print(f"  • month: Numeric month (1-12)")
if 'month_name' in df.columns:
    print(f"  • month_name: Month name (e.g., January, February)")
if 'time' in df.columns:
    print(f"  • time: Time in HH:MM:SS format")
if 'hour' in df.columns:
    print(f"  • hour: Hour of day (0-23)")
if 'top_level_domain' in df.columns:
    print(f"  • top_level_domain: TLD extracted from sender (e.g., com, org)")
if 'second_level_domain' in df.columns:
    print(f"  • second_level_domain: SLD extracted from sender (e.g., google, amazon)")
if 'subject_word_count' in df.columns:
    print(f"  • subject_word_count: Number of words in email subject")
if 'body_word_count' in df.columns:
    print(f"  • body_word_count: Number of words in email body")
if 'subject_avg_word_length' in df.columns:
    print(f"  • subject_avg_word_length: Average length of words in subject")
if 'body_avg_word_length' in df.columns:
    print(f"  • body_avg_word_length: Average length of words in body")

print("\n" + "=" * 60)
print("\nPreview of feature-engineered dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())