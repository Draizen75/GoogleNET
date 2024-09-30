import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import re
import joblib
import string
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt_tab')

st.set_page_config(layout='centered', page_title='Sentiment Analysis')

# Navigation buttons at the top
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home"):
        st.switch_page("Home.py")

with col2:
    if st.button("Kmeans Clustering"):
        st.switch_page("pages/KMEANS.py")

with col3:
    if st.button("Association Rules"):
        st.switch_page("pages/Association_Rules.py")

with col4:
    if st.button("Sentiment Analysis"):
        st.switch_page("pages/Sentiment_Analysis.py")


st.title("Sentiment Analysis")

# Load datasets
df_implementation = pd.read_csv('data/test.csv', encoding='Windows-1252')

@st.cache_data
# Function to handle sentiment analysis for the uploaded dataset
def sentiment_analysis(df_implementation):
    # Naive Bayes trained model
    vectorizer = joblib.load('model/vectorizer.joblib') # Loads custom countVectorizer
    model = joblib.load('model/Naive_Bayes_model.joblib')# Loads trained model
    
    # Predictions for the implementation dataset
    X_implementation = vectorizer.transform(df_implementation['comment'])
    df_implementation['predicted_sentiment'] = model.predict(X_implementation)

    # Get confidence scores
    proba_implementation = model.predict_proba(X_implementation)
    confidence_scores = proba_implementation.max(axis=1)  # Confidence of the predicted class
    df_implementation['accuracy'] = confidence_scores * 100  # Convert to percentage
    
    # Calculate classification report and accuracy score
    y_true = df_implementation['sentiment']
    y_pred = df_implementation['predicted_sentiment']
    
    accuracy = accuracy_score(y_true, y_pred)

    return df_implementation, vectorizer, model, accuracy, y_true, y_pred

def pre_process(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s0-9]', '', text) # Remove special char and numbers
    text = re.sub(r'\s+', ' ', text).strip()# Remove extra whitespace
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d', '', text)   # Remove digits
    cleaned_text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return cleaned_text

def count_words_by_sentiment(df, vectorizer):
    # Transform comments to token counts
    X = vectorizer.transform(df['comment'])
    
    # Create a DataFrame from the transformed data
    word_counts_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Add sentiment column
    word_counts_df['sentiment'] = df['sentiment']
    
    # Initialize counters
    positive_counts = word_counts_df[word_counts_df['sentiment'] == 'positive'].drop(columns='sentiment').sum()
    negative_counts = word_counts_df[word_counts_df['sentiment'] == 'negative'].drop(columns='sentiment').sum()
    
    return positive_counts, negative_counts

# Function to display output
def display_output(df):
    st.write(df)

# Function to predict sentiment for user input
def predict_sentiment(user_comment, vectorizer, model):
    user_comment_transformed = vectorizer.transform([user_comment])
    predicted_sentiment = model.predict(user_comment_transformed)[0]
    sentiment_accuracy = model.predict_proba(user_comment_transformed).max(axis=1)[0] * 100

    return predicted_sentiment, sentiment_accuracy

# Function to display classification report
def display_classification_report(y_true, y_pred):
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, labels=['positive', 'negative'], output_dict=True)
    st.dataframe(report)

    # Display Accuracy score
    st.write(f"**Accuracy Score:** {accuracy:.2f}")

# display Confusion matrix
def display_confusion_matrix(y_true, y_pred):
     # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative'])
    
    # Normalize the confusion matrix by row (true classes) to get percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    st.subheader("Confusion Matrix")
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    # Plot the confusion matrix counts
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])
    disp.plot(cmap='PuBuGn', ax=ax, values_format='d')
    
    # Overlay percentage values below the confusion matrix counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Add percentage text just below the count
            percentage_text = f'{cm_percentage[i, j]:.2f}%'
            # Adjust the positioning: count at the center, percentage slightly below
            ax.text(j, i + 0.2, percentage_text, ha='center', va='center', color='black', fontsize=8)  # Percentages

    # Customize plot appearance
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Render the plot in Streamlit
    st.pyplot(fig)
    
# Perform sentiment analysis
result_df, vectorizer, model, accuracy, y_true, y_pred = sentiment_analysis(df_implementation)

# Count words for positive and negative sentiments
positive_word_counts, negative_word_counts = count_words_by_sentiment(result_df, vectorizer)

# Convert word counts to DataFrames for better visualization
word_count_positive_df = pd.DataFrame(positive_word_counts.items(), columns=['Word', 'Count'])
word_count_negative_df = pd.DataFrame(negative_word_counts.items(), columns=['Word', 'Count'])

# Sort and select the top 20 words for each sentiment class
top_positive_words = word_count_positive_df.sort_values(by='Count', ascending=False).head(20)
top_negative_words = word_count_negative_df.sort_values(by='Count', ascending=False).head(20)

# Display Dataset, Data reports, and user input sentiment prediction
tab1, tab2, tab3 = st.tabs(['Dataset','Data Report', 'Try Sentiment Prediction'])

with tab1:
    # Display the analysis result
    display_output(result_df)
    
with tab2, st.container(border=True):
    col1, col2, col3 = st.columns(3)
    container = st.container(border=True)

    # Display classification report, accuracy score, and confusion matrix
    display_confusion_matrix(y_true, y_pred)
    colu1, colu2 = st.columns(2)
    with colu1:
        st.subheader("Top 20 Words - Positive Sentiment:")
        st.write()
        # Apply background gradient styling and display the DataFrame
        styled_positive_words = top_positive_words.style.background_gradient(cmap='Greens')
        styled_positive_words
    with colu2:
        # Display the top words for each sentiment class - Negative
        st.subheader("\nTop 20 Words - Negative Sentiment:")
        st.write()
        # Apply background gradient styling and display the DataFrame
        styled_negative_words = top_negative_words.style.background_gradient(cmap='Reds')
        styled_negative_words

    display_classification_report(y_true, y_pred)
    
with tab3:
    # User input for live sentiment prediction
    st.markdown("### Predict Sentiment for Your Own Comment")
    user_comment = st.text_input("Enter a comment:").lower()
    MIN_WORD_COUNT = 1
    if user_comment:
        cleaned_comment = pre_process(user_comment)
        word_count = len(cleaned_comment.split())  # Count words in cleaned comment

        if word_count < MIN_WORD_COUNT:
            st.warning(f"Please enter at least {MIN_WORD_COUNT} words.")
        else:
            predicted_sentiment, accuracy = predict_sentiment(cleaned_comment, vectorizer, model)
            st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
            st.write(f"**Accuracy:** {accuracy:.2f}%")
