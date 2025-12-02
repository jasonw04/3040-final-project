import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

def clean_text(text: str) -> str:
    """Basic cleaning: lowercase and keep only letters + spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model_and_vectorizer():
    """Load trained Logistic Regression model and TF-IDF vectorizer."""
    with open("logreg_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_data
def load_results_table():
    """Load model comparison results table."""
    return pd.read_csv("model_results_summary_sample200k_with_length_and_knn.csv")

def main():
    st.title("Amazon Reviews Sentiment – Project Demo")

    st.write(
        """
        This app showcases our final project on the **Amazon product review sentiment** data set.
        We compare several classifiers trained on TF–IDF features and review length,
        and let you try the final Logistic Regression model on your own text.
        """
    )

    tab_overview, tab_eda, tab_models, tab_cm, tab_predict = st.tabs(
        ["Overview", "EDA Visuals", "Model Results",
         "Confusion Matrices", "Try the Model"]
    )

    with tab_overview:
        st.header("Project Overview")

        st.markdown(
            """
            **Dataset Information**

            - Kaggle: *Amazon Reviews for Sentiment Analysis*  
            - Labels: `__label__1` (negative, 1–2 stars) and `__label__2` (positive, 4–5 stars)  
            - Full dataset: 3.6M reviews (balanced).  
            - For modelling we used a **sample of 200,000 reviews**.

            **Features**

            - TF–IDF for up to 20,000 unigrams / bigrams from the cleaned review text  
            - **Review length** (word count) as one extra numeric feature
            """
        )

        st.subheader("Reviewing Length by Sentiment")
        st.write(
            "Negative reviews tend to be longer and more variable, "
            "but there is a lot of overlap, so length alone is a weak predictor."
        )
        st.image("review_length_by_sentiment.png", caption="Review Length by Sentiment")

    with tab_eda:
        st.header("Exploratory Data Analysis – Review Length")

        st.subheader("Boxplot – Review Lengths (All Reviews)")
        st.image("eda_review_length_boxplot.png")

        st.subheader("Histogram – Review Lengths")
        st.image("eda_review_length_hist.png")

        st.subheader("QQ Plot – Review Lengths")
        st.image("eda_review_length_qq.png")

        st.subheader("Review Length by Sentiment (Seaborn Boxplot)")
        st.image("review_length_by_sentiment.png")

        st.subheader("Alternate Layout – Review Length by Sentiment")
        st.image("length_by_sentiment.png")

        st.caption(
            "Overall, review lengths are right-skewed: most reviews are relatively short, "
            "with a long tail of very long reviews."
        )

    with tab_models:
        st.header("Model Comparison")

        try:
            results_df = load_results_table()
            st.dataframe(results_df, use_container_width=True)
        except FileNotFoundError:
            st.warning(
                "Could not find 'model_results_summary_sample200k_with_length_and_knn.csv'. "
                "Place it in the same folder as this app."
            )

        st.markdown(
            """
            **Summary of performance**

            - **Logistic Regression**: best overall; ~89–90% accuracy and highest AUC.  
            - **Naive Bayes**: strong performance on sparse TF–IDF text.  
            - **Decision Tree**: reasonable but tends to overfit in high dimensions.  
            - **KNN**: struggles with distance in 20k-dimensional sparse space.  
            - **Baseline dummy classifier**: ~50% accuracy (majority class).
            """
        )

    with tab_cm:
        st.header("Confusion Matrices")

        st.subheader("Logistic Regression")
        st.image("cm_logistic_regression.png")

        st.subheader("Naive Bayes")
        st.image("cm_naive_bayes.png")

        st.subheader("Decision Tree")
        st.image("cm_decision_tree.png")

        st.subheader("K-Nearest Neighbors (KNN)")
        st.image("cm_knn.png")

        st.caption(
            "Logistic Regression and Naive Bayes have the lowest counts of false positives "
            "and false negatives, showing the clearest separation between positive and "
            "negative sentiment."
        )

    with tab_predict:
        st.header("Try the Logistic Regression Model")

        example_text = (
            "This product was amazing, worked perfectly and I would highly recommend it!"
        )
        review_text = st.text_area(
            "Enter your own review text to try the model!",
            value=example_text,
            height=150,
        )

        if st.button("Predict sentiment"):
            if not review_text.strip():
                st.warning("Please enter some text first.")
            else:
                try:
                    model, vectorizer = load_model_and_vectorizer()
                except FileNotFoundError:
                    st.error(
                        "Could not find 'logreg_model.pkl' and/or 'tfidf_vectorizer.pkl'. "
                        "Train the model offline and save these files in the same folder "
                        "as this app."
                    )
                    return

                # Preprocess text
                cleaned = clean_text(review_text)
                length = len(cleaned.split())

                # Vectorize text and append review length as last feature
                X_tfidf = vectorizer.transform([cleaned])
                length_arr = np.array([[length]])
                length_sparse = csr_matrix(length_arr)
                X_final = hstack([X_tfidf, length_sparse])

                proba_pos = model.predict_proba(X_final)[0, 1]
                label = "Positive" if proba_pos >= 0.5 else "Negative"

                st.markdown(f"### Predicted Sentiment: **{label}**")
                st.write(f"Probability of positive review: `{proba_pos:.3f}`")

                st.caption(
                    "The model uses thousands of TF–IDF word/bigram features plus review length. "
                    "Words like *excellent, amazing, love* push the probability up, while "
                    "*terrible, waste, refund* push it down."
                )


if __name__ == "__main__":
    main()
