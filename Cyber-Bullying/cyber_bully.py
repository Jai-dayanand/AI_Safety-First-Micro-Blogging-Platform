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

def train_aggression_model():
    csv_path = "cyber_bully.csv"  
    df = pd.read_csv(csv_path)
    required_columns = ['Text', 'ed_label_0', 'ed_label_1', 'oh_label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain the following columns: {required_columns}")
    def preprocess_text(text):
        text = text.lower()  # Lowercase
        text = re.sub(f"[{string.punctuation}]", "", text) 
        text = re.sub(r"\d+", "", text)  
        text = re.sub(r"\s+", " ", text).strip()  
        return text
    df['Text'] = df['Text'].apply(preprocess_text)
    X = df[['Text', 'ed_label_0', 'ed_label_1']]
    y = df['oh_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    text_transformer = TfidfVectorizer(max_features=5000)
    numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'Text'),
            ('numeric', numeric_transformer, ['ed_label_0', 'ed_label_1'])
        ]
    )
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model