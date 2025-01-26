import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV file
file_path = "/home/jashupadhyay/Documents/Repos/AI_Safety-First-Micro-Blogging-Platform/Cyber-Bullying/cyber_bully.csv"
df = pd.read_csv(file_path)

# Display dataset information (optional)
print("Dataset loaded successfully!")
print(df.head())
print("\nColumns:", df.columns)

# Features and labels
X_text = df["Text"]  # Text column
X_numeric = df[["ed_label_0", "ed_label_1"]]  # Numeric features
y = df["oh_label"]  # Target label

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_text_features = vectorizer.fit_transform(X_text)

# Combine text and numeric features
import numpy as np
X_combined = np.hstack([X_text_features.toarray(), X_numeric])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
