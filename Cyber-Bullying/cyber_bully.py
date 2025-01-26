from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
file_path = "/home/jashupadhyay/Documents/Repos/AI_Safety-First-Micro-Blogging-Platform/Cyber-Bullying/cyber_bully.csv"
df = pd.read_csv(file_path)
X_text = df["Text"]
X_numeric = df[["ed_label_0", "ed_label_1"]]
y = df["oh_label"]
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)
X_text_features = vectorizer.fit_transform(X_text)
X_combined = hstack([X_text_features, X_numeric])
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
