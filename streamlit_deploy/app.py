"""
app.py — AI-Generated Review Detector (Streamlit frontend)

Run locally:
    streamlit run app.py

Folder layout expected:
    app.py
    requirements.txt
    artifacts/
        word_vectorizer.joblib
        char_vectorizer.joblib
        style_scaler.joblib
        linear_svm.joblib
        logistic_regression.joblib
        xgboost.joblib
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AI-Generated Review Detector",
    page_icon="🕵️",
    layout="centered",
)

# =========================================================
# Load model artifacts (cached so it only loads once)
# =========================================================
import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

@st.cache_resource
def load_artifacts():
    word_vec = joblib.load(f"{ARTIFACTS_DIR}/word_vectorizer.joblib")
    char_vec = joblib.load(f"{ARTIFACTS_DIR}/char_vectorizer.joblib")
    scaler = joblib.load(f"{ARTIFACTS_DIR}/style_scaler.joblib")
    models = {
        "Linear SVM": joblib.load(f"{ARTIFACTS_DIR}/linear_svm.joblib"),
        "Logistic Regression": joblib.load(f"{ARTIFACTS_DIR}/logistic_regression.joblib"),
        "XGBoost": joblib.load(f"{ARTIFACTS_DIR}/xgboost.joblib"),
    }
    return word_vec, char_vec, scaler, models

word_vec, char_vec, scaler, models = load_artifacts()

LABELS = ["OR (Human-written)", "CG (AI-generated)"]

# =========================================================
# Feature engineering (must match training exactly)
# =========================================================
def stylometric_features(text_series):
    feats = pd.DataFrame(index=text_series.index)
    feats["char_len"] = text_series.str.len()
    feats["word_count"] = text_series.str.split().apply(len)
    feats["avg_word_len"] = feats["char_len"] / feats["word_count"].replace(0, 1)
    feats["punct_ratio"] = text_series.apply(lambda t: sum(1 for c in t if c in ".,!?;:") / max(len(t), 1))
    feats["upper_ratio"] = text_series.apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))
    feats["exclaim_count"] = text_series.str.count("!")
    return feats

def predict(text: str, model):
    s = pd.Series([text])
    Xw = word_vec.transform(s)
    Xc = char_vec.transform(s)
    Xs = scaler.transform(stylometric_features(s))
    X = hstack([Xw, Xc, Xs]).tocsr()

    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(X)[0, 1]
    elif hasattr(model, "decision_function"):
        conf = 1 / (1 + np.exp(-model.decision_function(X)[0]))
    else:
        conf = float(pred)
    return LABELS[pred], conf

# =========================================================
# UI
# =========================================================
st.title("🕵️ AI-Generated Review Detector")
st.write(
    "Paste a product review below to check whether it looks **human-written** "
    "or **AI-generated**."
)

model_choice = st.selectbox("Model", list(models.keys()), index=0)
model = models[model_choice]

review_text = st.text_area(
    "Review text",
    height=150,
    placeholder="e.g. This product exceeded my expectations! Highly recommend to anyone looking for a reliable option.",
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze:
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        label, conf = predict(review_text, model)
        is_ai = "CG" in label

        st.divider()
        if is_ai:
            st.error(f"**Prediction: {label}**")
        else:
            st.success(f"**Prediction: {label}**")

        st.metric("Confidence (probability of AI-generated)", f"{conf:.1%}")
        st.progress(min(max(conf, 0.0), 1.0))

        with st.expander("What does this mean?"):
            st.write(
                "The confidence score is the model's estimated probability that "
                "the review was AI-generated. Scores near 50% mean the model is "
                "unsure; scores near 0% or 100% indicate high confidence in the "
                "human-written or AI-generated prediction respectively."
            )

st.divider()
st.caption(
    "Model trained on the Amazon-style fake reviews dataset (40k+ labeled reviews) "
    "using TF-IDF (word + character n-grams) and stylometric features. "
    "Not guaranteed to be accurate on out-of-domain text — this is an academic project, not a production moderation tool."
)

# =========================================================
# Sidebar — try example reviews
# =========================================================
with st.sidebar:
    st.header("Try an example")
    examples = {
        "Likely AI-generated": "I recently purchased this item and I am extremely satisfied with the quality and performance. It exceeded my expectations and I would highly recommend it to anyone in the market for this type of product.",
        "Likely human-written": "took forever to arrive bc of some shipping delay but once it got here it worked fine. my cat knocked it off the counter on day 2 and it still works so thats a plus lol",
        "Borderline / short": "Does what it says. No complaints.",
    }
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["example_text"] = text

    if "example_text" in st.session_state:
        st.info("Example copied below — paste it into the review box above.")
        st.code(st.session_state["example_text"], language=None)

    st.divider()
    st.caption("AI-Generated Review Detector — FYP/portfolio project")
