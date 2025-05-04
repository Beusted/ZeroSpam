import streamlit as st
from model import load_model
from preprocessing import preprocess
import numpy as np

# Load model and vectorizer
model, vectorizer = load_model()


def get_top_keywords(text, vectorizer, model, top_n=5):
    features = vectorizer.get_feature_names_out()
    vec = vectorizer.transform([text])
    class_index = model.predict(vec)[0]
    coefs = model.feature_log_prob_[class_index]
    top_indices = np.argsort(vec.toarray()[0])[::-1]
    top_words = [features[i] for i in top_indices if vec.toarray()[0][i] > 0][:top_n]
    return top_words


# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")

user_input = st.text_area("Enter your email text below:", height=200)

if st.button("Classify"):
    if user_input.strip():
        clean_input = preprocess(user_input)
        input_vector = vectorizer.transform([clean_input])
        prediction = model.predict(input_vector)[0]
        confidence = round(np.max(model.predict_proba(input_vector)) * 100, 2)

        result = "🟥 SPAM" if prediction == 1 else "🟩 NOT SPAM"
        st.subheader("Result:")
        st.success(f"{result} (Confidence: {confidence}%)")

        keywords = get_top_keywords(clean_input, vectorizer, model)
        st.markdown(f"**Top Influential Words:** {', '.join(keywords)}")
    else:
        st.warning("Please enter some text.")
