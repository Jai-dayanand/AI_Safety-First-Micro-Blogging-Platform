import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import re
import string


def train_aggression_model(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_columns = ['Text', 'ed_label_0', 'ed_label_1', 'oh_label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain the following columns: {required_columns}")

    # Text preprocessing function
    def preprocess_text(text):
        text = text.lower()  # Lowercase
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    # Preprocess the Text column
    df['Text'] = df['Text'].apply(preprocess_text)

    # Features and labels
    X = df[['Text', 'ed_label_0', 'ed_label_1']]
    y = df['oh_label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessor for numeric and text features
    text_transformer = TfidfVectorizer(max_features=5000)
    numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'Text'),
            ('numeric', numeric_transformer, ['ed_label_0', 'ed_label_1'])
        ]
    )

    # Create a pipeline with Logistic Regression
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    # Example usage
    csv_path = "cyber_bully.csv"  # Replace with the path to your CSV
    model = train_aggression_model(csv_path)

    # Testing the model on custom input
    sample_data = pd.DataFrame({
        'Text': ["You are amazing!", "I hate this!"],
        'ed_label_0': [0.9, 0.1],
        'ed_label_1': [0.1, 0.9]
    })
    predictions = model.predict(sample_data)
    print("Predictions for sample data:", predictions)
