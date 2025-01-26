from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_cyber_bullying_model(file_path):
    """
    Trains a logistic regression model for cyberbullying detection.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: A trained model, vectorizer, and feature column names for numerical data.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Separate features and labels
    X_text = df["Text"]
    X_numeric = df[["ed_label_0", "ed_label_1"]]
    y = df["oh_label"]

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)
    X_text_features = vectorizer.fit_transform(X_text)

    # Combine text features with numeric features
    X_combined = hstack([X_text_features, X_numeric])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, vectorizer, ["ed_label_0", "ed_label_1"]

def predict_cyber_bullying(model, vectorizer, numeric_columns, texts, numeric_features):
    """
    Predicts whether the given texts are bullying or not.

    Args:
        model: Trained logistic regression model.
        vectorizer: Fitted TfidfVectorizer.
        numeric_columns (list): List of numeric feature column names.
        texts (list): List of text inputs to predict.
        numeric_features (pd.DataFrame): DataFrame containing numeric features corresponding to the texts.

    Returns:
        str: Single prediction as "bully" or "not_bully".
    """
    # Vectorize the input texts
    X_text_features = vectorizer.transform(texts)

    # Combine text features with numeric features
    X_combined = hstack([X_text_features, numeric_features])

    # Make a single prediction
    prediction = model.predict(X_combined)[0]

    # Convert prediction to "bully" or "not_bully"
    return "bully" if prediction == 1 else "not_bully"

# Example usage:
if __name__ == "__main__":
    # Path to the dataset
    file_path = "/home/jashupadhyay/Documents/Repos/AI_Safety-First-Micro-Blogging-Platform/Cyber-Bullying/cyber_bully.csv"

    # Train the model
    model, vectorizer, numeric_columns = train_cyber_bullying_model(file_path)

    # Example inputs for prediction
    sample_text = "You are amazing!"
    sample_numeric_features = pd.DataFrame(
        {"ed_label_0": [0], "ed_label_1": [0.2]}
    )

    # Make a prediction
    prediction = predict_cyber_bullying(
        model, vectorizer, numeric_columns, [sample_text], sample_numeric_features
    )

    print("Prediction:", prediction)
