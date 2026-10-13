
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.title("AI-Based Spam Mail Detection")
st.write("Enter an email message to check whether it is spam.")

@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv")

    data = data[["label", "message"]].dropna()
    data["label"] = data["label"].str.lower().str.strip()
    data["message"] = data["message"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test, model.predict(X_test)
    )

    return model, accuracy


if st.button("Load AI Model"):
    try:
        model, accuracy = train_model()
        st.session_state["model"] = model
        st.session_state["accuracy"] = accuracy
        st.success("AI model trained successfully!")
    except Exception as error:
        st.error(f"Error loading model: {error}")

if "model" in st.session_state:
    st.write(
        f"Test accuracy: {st.session_state['accuracy']:.1%}"
    )

    message = st.text_area(
        "Paste your email message here:"
    )

    if st.button("Detect Spam"):
        if message.strip():
            prediction = st.session_state["model"].predict(
                [message]
            )[0]

            if prediction == "spam":
                st.error("⚠️ This message is classified as SPAM.")
            else:
                st.success(
                    "✅ This message is classified as NOT SPAM."
                )
        else:
            st.warning("Please enter a message.")
