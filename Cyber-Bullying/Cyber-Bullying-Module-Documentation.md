# Cyberbullying Detection Project

This project aims to build a **Cyberbullying Detector** using publicly available datasets and a Logistic Regression model. Below is an overview of the datasets used and the approach being implemented.

## Dataset Sources
The datasets used for this project were taken from the [Kaggle Cyberbullying Dataset](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?select=twitter_sexism_parsed_dataset.csv). These datasets encompass a variety of harmful online behaviors:

1. **Aggression**
   - Focuses on identifying aggressive language in online interactions.
2. **Attack**
   - Contains samples of personal attacks targeted at individuals or groups.
3. **Racism**
   - Includes tweets with racial slurs or offensive comments based on race.
4. **Sexism**
   - Contains examples of misogynistic or sexist comments.
5. **Toxicity**
   - Covers general toxic behavior, including abusive or profane language.

## Objective
The primary goal is to classify online text (e.g., tweets, comments) into one or more of the above harmful categories to mitigate the negative impact of cyberbullying.

## Methodology
1. **Data Preprocessing**
   - Data cleaning: Removing special characters, URLs, and stopwords.
   - Text vectorization: Converting text into numerical representations using techniques such as TF-IDF or Count Vectorization.

2. **Modeling with Logistic Regression**
   - Logistic Regression will be employed as the primary model for classification.
   - A one-vs-rest (OvR) approach will be used to handle multi-class classification, where a separate logistic regression model is trained for each class.

3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, and F1-score for each category
   - Confusion matrix for detailed class-wise performance analysis

## Why Logistic Regression?
Logistic Regression is a robust and interpretable algorithm for binary and multi-class classification. Its simplicity, coupled with efficiency for text data, makes it an excellent choice for this project.

## Expected Outcomes
- A trained Logistic Regression model capable of detecting cyberbullying across the five categories.
- Insights into the most impactful words and phrases contributing to harmful behavior classification.
- A foundation for future enhancements, such as using deep learning or integrating real-time detection in social media applications.


node --max-old-space-size=25600 Cyber-Bullying/Sexism_Classifier.js