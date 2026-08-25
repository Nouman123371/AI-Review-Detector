"""
app.py — AI-Generated Review Detector
Professional Streamlit Frontend

Run:
    streamlit run app.py

Required artifacts:
    artifacts/
        word_vectorizer.joblib
        char_vectorizer.joblib
        style_scaler.joblib
        linear_svm.joblib
        logistic_regression.joblib
        xgboost.json
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import hstack
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Review Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 30%),
            #0b1020;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */

    .hero {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(30,41,59,0.95),
            rgba(15,23,42,0.95)
        );
        border: 1px solid rgba(148,163,184,0.15);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        margin-bottom: 25px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(129,140,248,0.3);
        color: #a5b4fc;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 17px;
        margin-top: 10px;
        line-height: 1.6;
    }

    /* ---------- Cards ---------- */

    .card {
        padding: 22px;
        border-radius: 20px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.12);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .card-description {
        color: #94a3b8;
        font-size: 14px;
    }

    /* ---------- Result ---------- */

    .result-ai {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(127,29,29,0.35),
            rgba(30,41,59,0.8)
        );
        border: 1px solid rgba(248,113,113,0.35);
        text-align: center;
        margin: 20px 0;
    }

    .result-human {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(20,83,45,0.35),
            rgba(30,41,59,0.8)
        );
        border: 1px solid rgba(74,222,128,0.35);
        text-align: center;
        margin: 20px 0;
    }

    .result-icon {
        font-size: 45px;
    }

    .result-label {
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .result-confidence {
        color: #cbd5e1;
        margin-top: 5px;
    }

    /* ---------- Stats ---------- */

    .stat-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(30,41,59,0.6);
        border: 1px solid rgba(148,163,184,0.1);
        text-align: center;
    }

    .stat-number {
        font-size: 25px;
        font-weight: 750;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 4px;
    }

    /* ---------- Info ---------- */

    .info-box {
        padding: 18px;
        border-radius: 16px;
        background: rgba(30,41,59,0.55);
        border-left: 4px solid #818cf8;
        color: #cbd5e1;
        line-height: 1.6;
    }

    .warning-box {
        padding: 18px;
        border-radius: 16px;
        background: rgba(120,53,15,0.22);
        border: 1px solid rgba(251,191,36,0.25);
        color: #fde68a;
        line-height: 1.6;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148,163,184,0.1);
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        min-height: 45px;
    }

    /* ---------- Text Area ---------- */

    textarea {
        border-radius: 15px !important;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATHS
# =========================================================

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts"
)


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():

    word_vec = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "word_vectorizer.joblib"
        )
    )

    char_vec = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "char_vectorizer.joblib"
        )
    )

    scaler = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "style_scaler.joblib"
        )
    )

    linear_svm = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "linear_svm.joblib"
        )
    )

    logistic_regression = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "logistic_regression.joblib"
        )
    )

    # XGBoost native JSON model
    xgb_model = XGBClassifier()
    xgb_model.load_model(
        os.path.join(
            ARTIFACTS_DIR,
            "xgboost.json"
        )
    )

    models = {
        "Linear SVM": linear_svm,
        "Logistic Regression": logistic_regression,
        "XGBoost": xgb_model,
    }

    return word_vec, char_vec, scaler, models


word_vec, char_vec, scaler, models = load_artifacts()


# =========================================================
# LABELS
# =========================================================

LABELS = [
    "OR (Human-written)",
    "CG (AI-generated)"
]


# =========================================================
# STYLometric FEATURES
# =========================================================

def stylometric_features(text_series):

    feats = pd.DataFrame(index=text_series.index)

    feats["char_len"] = text_series.str.len()

    feats["word_count"] = (
        text_series
        .str.split()
        .apply(len)
    )

    feats["avg_word_len"] = (
        feats["char_len"] /
        feats["word_count"].replace(0, 1)
    )

    feats["punct_ratio"] = text_series.apply(
        lambda t:
        sum(
            1 for c in t
            if c in ".,!?;:"
        ) / max(len(t), 1)
    )

    feats["upper_ratio"] = text_series.apply(
        lambda t:
        sum(
            1 for c in t
            if c.isupper()
        ) / max(len(t), 1)
    )

    feats["exclaim_count"] = (
        text_series
        .str.count("!")
    )

    return feats


# =========================================================
# PREDICTION
# =========================================================

def predict(text, model):

    s = pd.Series([text])

    Xw = word_vec.transform(s)

    Xc = char_vec.transform(s)

    Xs = scaler.transform(
        stylometric_features(s)
    )

    X = hstack(
        [Xw, Xc, Xs]
    ).tocsr()

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

        # Class 1 = AI generated
        conf_ai = float(probabilities[1])

    elif hasattr(model, "decision_function"):

        decision = float(
            model.decision_function(X)[0]
        )

        conf_ai = float(
            1 / (1 + np.exp(-decision))
        )

    else:

        conf_ai = float(pred)

    if pred == 1:
        label = LABELS[1]
    else:
        label = LABELS[0]

    return label, conf_ai


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center">

        <div style="font-size:55px">🕵️</div>

        <h2 style="margin-bottom:0">
        AI Review Detector
        </h2>

        <p style="color:#94a3b8">
        ML-powered review authenticity analysis
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Detection Model")

    model_choice = st.selectbox(
        "Choose model",
        list(models.keys()),
        index=0,
        help="Select the machine learning model used for prediction."
    )

    model_descriptions = {

        "Linear SVM":
            "Fast and effective linear classifier.",

        "Logistic Regression":
            "Probability-based linear classification model.",

        "XGBoost":
            "Gradient boosting model designed for strong classification performance."
    }

    st.caption(
        model_descriptions[model_choice]
    )

    st.divider()

    st.markdown("### 🧪 Try Sample Reviews")

    examples = {

        "🤖 AI-style review":
            "I recently purchased this item and I am extremely satisfied with the quality and performance. It exceeded my expectations and I would highly recommend it to anyone looking for a reliable product.",

        "👤 Human-style review":
            "took forever to arrive bc of some shipping delay but once it got here it worked fine. my cat knocked it off the counter on day 2 and it still works so thats a plus lol",

        "⚖️ Short / borderline":
            "Does what it says. No complaints."
    }

    for label, text in examples.items():

        if st.button(
            label,
            use_container_width=True
        ):

            st.session_state[
                "review_text"
            ] = text

    st.divider()

    st.markdown("### 📌 About")

    st.caption(
        """
        This academic project uses:

        • Word TF-IDF features  
        • Character TF-IDF features  
        • Stylometric features  
        • Multiple ML classifiers
        """
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            🔬 MACHINE LEARNING • NLP • TEXT ANALYSIS
        </div>

        <div class="hero-title">
            AI-Generated Review Detector
        </div>

        <div class="hero-subtitle">
            Analyze product reviews and estimate whether the text
            appears to be <b>human-written</b> or <b>AI-generated</b>.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP INFORMATION CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-number">
                📝
            </div>

            <div class="stat-label">
                Text Analysis
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-number">
                🧠
            </div>

            <div class="stat-label">
                NLP + ML
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-number">
                3
            </div>

            <div class="stat-label">
                ML Models
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-number">
                ⚡
            </div>

            <div class="stat-label">
                Instant Prediction
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# MAIN INPUT AREA
# =========================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            ✍️ Enter a Product Review
        </div>

        <div class="card-description">
            Paste a review below and our machine learning model
            will analyze its linguistic and stylometric patterns.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""


review_text = st.text_area(
    "Review",
    value=st.session_state["review_text"],
    height=220,
    label_visibility="collapsed",
    placeholder=(
        "Example: I bought this product last week and honestly "
        "I wasn't expecting much, but it works surprisingly well..."
    )
)


# =========================================================
# TEXT STATISTICS
# =========================================================

word_count = len(review_text.split())

character_count = len(review_text)

sentence_count = max(
    1,
    sum(
        review_text.count(x)
        for x in [".", "!", "?"]
    )
) if review_text.strip() else 0


c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Words",
        word_count
    )

with c2:
    st.metric(
        "Characters",
        character_count
    )

with c3:
    st.metric(
        "Sentences",
        sentence_count
    )


# =========================================================
# ACTION BUTTONS
# =========================================================

button1, button2, button3 = st.columns(
    [2, 1, 1]
)

with button1:

    analyze = st.button(
        "🔍 Analyze Review",
        type="primary",
        use_container_width=True
    )

with button2:

    clear = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

with button3:

    if st.button(
        "🎲 Random Example",
        use_container_width=True
    ):

        random_text = np.random.choice(
            list(examples.values())
        )

        st.session_state[
            "review_text"
        ] = random_text

        st.rerun()


if clear:

    st.session_state[
        "review_text"
    ] = ""

    st.rerun()


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not review_text.strip():

        st.warning(
            "⚠️ Please enter a review before analyzing."
        )

    elif len(review_text.strip()) < 10:

        st.warning(
            "⚠️ The review is very short. "
            "Longer text generally provides more information "
            "for the model."
        )

    else:

        with st.spinner(
            "🧠 Analyzing linguistic patterns..."
        ):

            label, conf_ai = predict(
                review_text,
                models[model_choice]
            )

        conf_ai = float(
            np.clip(
                conf_ai,
                0,
                1
            )
        )

        conf_human = 1 - conf_ai

        is_ai = (
            "CG" in label
        )


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        if is_ai:

            st.markdown(
                f"""
                <div class="result-ai">

                    <div class="result-icon">
                        🤖
                    </div>

                    <div class="result-label">
                        AI-Generated
                    </div>

                    <div class="result-confidence">
                        The model estimates a
                        <b>{conf_ai:.1%}</b>
                        probability of AI-generated text.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-human">

                    <div class="result-icon">
                        👤
                    </div>

                    <div class="result-label">
                        Human-Written
                    </div>

                    <div class="result-confidence">
                        The model estimates a
                        <b>{conf_human:.1%}</b>
                        probability of human-written text.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITY BREAKDOWN
        # =================================================

        st.markdown(
            "### 📊 Prediction Breakdown"
        )

        p1, p2 = st.columns(2)

        with p1:

            st.metric(
                "🤖 AI-Generated Probability",
                f"{conf_ai:.1%}"
            )

            st.progress(
                conf_ai
            )

        with p2:

            st.metric(
                "👤 Human-Written Probability",
                f"{conf_human:.1%}"
            )

            st.progress(
                conf_human
            )


        # =================================================
        # CONFIDENCE WARNING
        # =================================================

        if 0.40 <= conf_ai <= 0.60:

            st.markdown(
                """
                <div class="warning-box">

                ⚠️ <b>Borderline prediction</b><br><br>

                The model is relatively uncertain about this review.
                A probability close to 50% means the text contains
                characteristics that could be associated with either
                human or AI-generated writing.

                </div>
                """,
                unsafe_allow_html=True
            )


        elif conf_ai >= 0.80 or conf_ai <= 0.20:

            st.success(
                "The model has relatively high confidence in this prediction."
            )


        # =================================================
        # REVIEW ANALYSIS
        # =================================================

        st.markdown(
            "### 🔬 Review Analysis"
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.metric(
                "Words",
                word_count
            )

        with a2:

            st.metric(
                "Characters",
                character_count
            )

        with a3:

            avg_word_length = (
                character_count /
                max(word_count, 1)
            )

            st.metric(
                "Avg. Word Length",
                f"{avg_word_length:.1f}"
            )

        with a4:

            exclamations = review_text.count("!")

            st.metric(
                "Exclamation Marks",
                exclamations
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        with st.expander(
            "🧠 How does this detector work?"
        ):

            st.markdown(
                f"""
                **Selected model:** `{model_choice}`

                The detector combines multiple types of text
                characteristics:

                **1. Word-level TF-IDF**

                Identifies important word and phrase patterns.

                **2. Character-level TF-IDF**

                Captures writing patterns at the character level,
                including spelling, punctuation and word structure.

                **3. Stylometric features**

                The system also considers:

                - Character length
                - Word count
                - Average word length
                - Punctuation ratio
                - Uppercase ratio
                - Exclamation count

                These features are combined and passed to the
                selected machine learning classifier.
                """
            )


        # =================================================
        # DISCLAIMER
        # =================================================

        st.markdown(
            """
            <div class="info-box">

            💡 <b>Important:</b>

            This detector estimates whether text resembles patterns
            found in AI-generated or human-written reviews.

            It should not be treated as definitive proof of authorship.
            Short reviews, unusual writing styles, edited AI text,
            and text outside the training distribution may produce
            unreliable predictions.

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        <b>AI-Generated Review Detector</b><br>

        Academic / Portfolio Machine Learning Project<br>

        TF-IDF • Stylometric Features • Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
