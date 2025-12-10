# Step 0: Suppress TensorFlow and general warnings to keep output clean
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings

warnings.filterwarnings("ignore")  # Suppress all Python warnings

# Step 1: Import required libraries
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations
import re  # For regular expressions used in text cleaning

# Step 2: Import machine learning libraries
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.naive_bayes import GaussianNB  # Naive Bayes model
from xgboost import XGBClassifier  # XGBoost model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)

# Step 3: Import deep learning libraries
from tensorflow.keras.models import Sequential  # Sequential model builder
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D  # Layers for LSTM model
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenizer for text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Padding sequences to equal length

# Step 4: Import BERT libraries
from transformers import logging

logging.set_verbosity_error()  # Suppress transformer warnings

# Step 5: Load the cleaned phishing dataset
MAX_ROWS = 1500
df = pd.read_csv('Cleaned_PhishingEmailData.csv', encoding='ISO-8859-1',
                 nrows=MAX_ROWS)  # Load CSV with proper encoding


# Step 6: Feature engineering from email content
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "link", text)  # Replace URLs with 'link'
    text = re.sub(r"\S+@\S+", "email", text)  # Replace email addresses with 'email'
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    return text


# Apply text cleaning and extract features
df['clean_content'] = df['body'].apply(clean_text)
df['subject_length'] = df['subject'].apply(lambda x: len(str(x)))  # Length of subject line
df['link_count'] = df['body'].apply(lambda x: len(re.findall(r"http\S+", str(x))))  # Number of links
df['has_link'] = df['link_count'].apply(lambda x: 1 if x > 0 else 0)  # Binary indicator for links

# Step 7: Select features and target variable
features = ['subject_length', 'link_count', 'has_link']
X = df[features]  # Feature matrix
y = df['label']  # Target labels

# Step 8: Split data into training and testing sets and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 9: Define evaluation function to compute metrics
def evaluate_model(y_true, y_pred):
    y_true_oh = pd.get_dummies(y_true)  # One-hot encode true labels
    y_pred_oh = pd.get_dummies(y_pred)  # One-hot encode predicted labels
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }


# Step 10: Define function to plot evaluation metrics
def plot_metrics(name, metrics):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='Blues')
    plt.title(f"{name} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Step 11: Train and evaluate classical models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}

results = {}  # Store evaluation results
predictions = {}  # Store predictions
# Loop through each classical model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Train model
    y_pred = model.predict(X_test_scaled)  # Predict on test set
    predictions[name] = y_pred
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))  # Print detailed report
    metrics = evaluate_model(y_test, y_pred)  # Compute metrics
    results[name] = metrics
    plot_metrics(name, metrics)  # Plot metrics

# Step 12: Train and evaluate LSTM model
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_content'])  # Fit tokenizer on cleaned text
X_seq = tokenizer.texts_to_sequences(df['clean_content'])  # Convert text to sequences
X_pad = pad_sequences(X_seq, maxlen=100)  # Pad sequences to fixed length

# Split padded sequences
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_pad, y,
                                                                        test_size=0.2, random_state=42)
# Define LSTM model architecture
lstm_model = Sequential()
lstm_model.add(Embedding(5000, 128))  # Embedding layer
lstm_model.add(SpatialDropout1D(0.2))  # Dropout for regularisation
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
lstm_model.add(Dense(1, activation='sigmoid'))  # Output layer
lstm_model.compile(loss='binary_crossentropy', optimizer='adam',
                   metrics=['accuracy'])  # Compile model
# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=3, batch_size=64, verbose=0)
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()  # Predict and threshold
predictions['LSTM'] = y_pred_lstm
print("\nLSTM Classification Report:\n")
print(classification_report(y_test_lstm, y_pred_lstm, zero_division=0))
metrics_lstm = evaluate_model(y_test_lstm, y_pred_lstm)
results['LSTM'] = metrics_lstm
plot_metrics("LSTM", metrics_lstm)

# EXTRA MODEL

import pandas as pd
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure correct column names
TEXT_COL = "body"
LABEL_COL = "label"

print("Checking for missing values...")
print(df.isna().sum(), "\n")

# Convert labels to int (phishing=1, ham=0)
df[LABEL_COL] = df[LABEL_COL].astype(int)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL],
    df[LABEL_COL],
    test_size=0.2,
    random_state=42,
    stratify=df[LABEL_COL]
)

print("Dataset split complete!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}\n")

# --------------------------------------------------------------
# Sentence Transformer Embeddings
# --------------------------------------------------------------

print("🔍 Loading Sentence-Transformer model (this may take a moment)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

print("Encoding training emails...")
X_train_embeddings = model.encode(X_train.tolist(), show_progress_bar=True)
print("Training embeddings shape:", X_train_embeddings.shape, "\n")

print("Encoding test emails...")
X_test_embeddings = model.encode(X_test.tolist(), show_progress_bar=True)
print("Test embeddings shape:", X_test_embeddings.shape, "\n")

# --------------------------------------------------------------
# LightGBM Model
# --------------------------------------------------------------

print("🚀 Training LightGBM model...")

lgb_model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=300
)

lgb_model.fit(X_train_embeddings, y_train)

print("Training complete!\n")

# --------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------

print("📊 Making predictions...")
preds = lgb_model.predict(X_test_embeddings)
pred_probs = lgb_model.predict_proba(X_test_embeddings)

print("Predictions completed!\n")

print("🔎 Accuracy:")
print(accuracy_score(y_test, preds), "\n")

print("📝 Classification Report:")
print(classification_report(y_test, preds), "\n")

print("📦 Confusion Matrix:")
print(confusion_matrix(y_test, preds), "\n")

# --------------------------------------------------------------
# Example Predictions
# --------------------------------------------------------------

sample_emails = [
    "Your account has been locked, click here to unlock.",
    "Hey, here is the agenda for tomorrow’s meeting.",
    "You won a free reward. Claim it now!",
    "Invoice attached from yesterday."
]

print("🔮 Example Predictions:")
sample_embeddings = model.encode(sample_emails)

sample_preds = lgb_model.predict(sample_embeddings)
sample_probs = lgb_model.predict_proba(sample_embeddings)

for email, pred, prob in zip(sample_emails, sample_preds, sample_probs):
    print("\nEmail:", email)
    print("Prediction:", "Phishing" if pred == 1 else "Ham")
    print("Confidence:", round(max(prob), 3))

print("\n🎉 Done!")

# ---------- PYTORCH DISTILBERT (APPLE-SILICON SAFE) ----------

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_texts = df.loc[X_train.index, "body"].astype(str).tolist()
test_texts = df.loc[X_test.index, "body"].astype(str).tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)


# Torch datasets
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = torch.tensor(self.labels[idx])
        return item, label

    def __len__(self):
        return len(self.labels)


train_dataset = EmailDataset(train_encodings, y_train.values)
test_dataset = EmailDataset(test_encodings, y_test.values)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

optim = torch.optim.Adam(model.parameters(), lr=5e-5)

# Training loop (1–2 epochs only)
for epoch in range(2):
    model.train()
    for batch, labels in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)

        outputs = model(**batch, labels=labels)
        loss = outputs.loss

        optim.zero_grad()
        loss.backward()
        optim.step()

# Evaluation
model.eval()
preds = []
with torch.no_grad():
    for batch, labels in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

predictions["DistilBERT"] = preds

print("\nDistilBERT Classification Report:\n")
print(classification_report(y_test, preds, zero_division=0))

metrics_distil = evaluate_model(y_test, preds)
results["DistilBERT"] = metrics_distil
plot_metrics("DistilBERT", metrics_distil)
