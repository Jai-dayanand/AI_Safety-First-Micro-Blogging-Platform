import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re
import string


def train_aggression_model(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Text preprocessing function
    def preprocess_text(text):
        text = text.lower()  # Lowercase
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    # Preprocess the text column
    df['text'] = df['text'].apply(preprocess_text)

    # Split the data
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('logreg', LogisticRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Prediction function
    def predict_aggression(texts):
        """Predicts aggression for a list of texts."""
        processed_texts = [preprocess_text(text) for text in texts]
        return pipeline.predict(processed_texts)

    return pipeline, predict_aggression


# Main function for testing
def main():
    # Path to your CSV file
    csv_path = "cyber_bully.csv"  # Replace with the actual path to your CSV file

    # Train the model and get the prediction function
    model, predict_fn = train_aggression_model(csv_path)

    # Test the model with some sample inputs
    sample_texts = [
        "I hate you!",
        "Have a nice day!",
        "You're so stupid and annoying.",
        "What a wonderful world this is."
    ]
    predictions = predict_fn(sample_texts)

    # Display the results
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {'Aggression' if pred == 1 else 'Not Aggression'}")
        print("-" * 30)


if __name__ == "__main__":
    main()
