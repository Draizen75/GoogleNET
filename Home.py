import streamlit as st



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

st.title("DATA VISUALIZATION AND ANALYSIS")

st.write(
    """
    ## ANALYZING CUSTOMER PREFERENCES OF BICYCLE PARTS FROM SHOPEE USING DATA MINING
    
    Welcome to our study on analyzing customer preferences of bicycle parts from Shopee using data mining techniques. This project aims to uncover insights into customer behavior and preferences in the bicycle parts market on Shopee.
    
    Our analysis employs various data mining methods including K-means clustering, association rule mining, and sentiment analysis to understand customer preferences and trends in the bicycle parts and accessories market.
    
    The dataset used in this study, containing Shopee bicycle parts and accessories reviews, can be found [here](https://huggingface.co/datasets/lllaurenceee/Shopee_Bicycle_Reviews).
    """
)
