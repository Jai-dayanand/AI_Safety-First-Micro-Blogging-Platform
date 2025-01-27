import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import re


def bot_detection_pipeline(dataset_path, input_text, retweet_count, mention_count, follower_count, verified):
    """
    A single function for bot detection that includes the entire pipeline:
    dataset loading, preprocessing, model training, and prediction.

    Args:
        dataset_path (str): Path to the dataset CSV file.
        input_text (str): Tweet text for prediction.
        retweet_count (int): Retweet count for the tweet.
        mention_count (int): Mention count for the tweet.
        follower_count (int): Follower count of the user.
        verified (bool): Whether the user is verified or not.

    Returns:
        str: Prediction result ("BOT" or "NOT BOT").
    """
    # Step 1: Load Dataset
    df = pd.read_csv(dataset_path)

    # Check for column names and strip extra spaces
    df.columns = df.columns.str.strip()

    # Verify the existence of required columns
    required_columns = ['Tweet', 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Bot Label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The column '{col}' is not found in the dataset. Please verify the column names.")

    # Step 2: Preprocessing
    def preprocess_text(text):
        text = re.sub(r'\W', ' ', str(text))  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.lower()

    # Apply preprocessing to the Tweet column
    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    # Step 3: Select Relevant Features and Target
    X_text = df['Tweet']  # Text data
    X_numeric = df[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']].copy()  # Numeric features
    X_numeric['Verified'] = X_numeric['Verified'].astype(int)  # Convert Verified (True/False) to 1/0
    y = df['Bot Label']  # Target (bot or human)

    # Step 4: Split Data
    X_train_text, X_test_text, X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42
    )

    # Step 5: Vectorize Text Features (Limit the number of features)
    vectorizer = TfidfVectorizer(max_features=500)  # Use top 500 features instead of 1000
    X_train_text_vec = vectorizer.fit_transform(X_train_text)
    X_test_text_vec = vectorizer.transform(X_test_text)

    # Step 6: Standardize Numeric Features
    scaler = StandardScaler()
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
    X_test_numeric_scaled = scaler.transform(X_test_numeric)

    # Step 7: Combine Text and Numeric Features
    X_train_combined = hstack([X_train_text_vec, X_train_numeric_scaled])  # Combine text and numeric features
    X_test_combined = hstack([X_test_text_vec, X_test_numeric_scaled])

    # Step 8: Train Logistic Regression Model
    model = LogisticRegression(max_iter=500)  # Reduced iterations
    model.fit(X_train_combined, y_train)

    # Step 9: Evaluate Model
    y_pred = model.predict(X_test_combined)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

    # Step 10: Bot Detection
    # Preprocess text input
    input_text = preprocess_text(input_text)
    input_text_vec = vectorizer.transform([input_text])

    # Scale numeric features
    input_numeric = scaler.transform([[retweet_count, mention_count, follower_count, int(verified)]])

    # Combine text and numeric features
    input_combined = hstack([input_text_vec, input_numeric])

    # Predict
    prediction = model.predict(input_combined)
    return 1 if prediction[0] == 1 else 0