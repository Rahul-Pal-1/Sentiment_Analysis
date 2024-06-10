import streamlit as st
import pickle
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Function to read the uploaded file and return the comments
def read_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        comments = file_content.splitlines()
        return comments
    return []

# Load the models and vectorizer
@st.cache_data
def load_models_and_vectorizer():
    with open('../Models/model_dt.pkl', 'rb') as dt_file:
        model_dt = pickle.load(dt_file)
    with open('../Models/model_xgb.pkl', 'rb') as xgb_file:
        model_xgb = pickle.load(xgb_file)
    with open('../Models/model_rf.pkl', 'rb') as rf_file:
        model_rf = pickle.load(rf_file)
    with open('../Models/countVectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model_dt, model_xgb, model_rf, vectorizer

# Predict sentiment
def predict_sentiment(model_dt, model_xgb, model_rf, vectorizer, comments):
    comments_vectorized = vectorizer.transform(comments)
    predictions_dt = model_dt.predict(comments_vectorized)
    predictions_xgb = model_xgb.predict(comments_vectorized)
    predictions_rf = model_rf.predict(comments_vectorized)
    return predictions_dt, predictions_xgb, predictions_rf


def main():
    st.title("Comment Sentiment Analysis")

    st.write("### Enter your comments:")
    comments_input = st.text_area("Type your comments here (each comment on a new line):")

    st.write("### Or upload a file containing comments:")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])

    if uploaded_file is not None:
        comments = read_uploaded_file(uploaded_file)
    else:
        comments = comments_input.split("\n")
    
    if st.button("Analyze Sentiment"):
        model_dt, model_xgb, model_rf, vectorizer = load_models_and_vectorizer()
        if comments:
            predictions_dt, predictions_xgb, predictions_rf = predict_sentiment(model_dt, model_xgb, model_rf, vectorizer, comments)
            st.write("### Sentiment Analysis Results:")
            for comment, pred_dt, pred_xgb, pred_rf in zip(comments, predictions_dt, predictions_xgb, predictions_rf):
                dt_sentiment = "Positive" if pred_dt == 1 else "Negative"
                xgb_sentiment = "Positive" if pred_xgb == 1 else "Negative"
                rf_sentiment = "Positive" if pred_rf == 1 else "Negative"
                st.write(f"Comment: {comment}")
                st.write(f"Decision Tree Sentiment: {dt_sentiment}")
                st.write(f"Random Forest Sentiment: {rf_sentiment}")
                st.write("---")
        else:
            st.write("Please enter some comments or upload a file.")

if __name__ == "__main__":
    main()
