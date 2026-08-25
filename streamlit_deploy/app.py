"""
app.py — Professional AI-Generated Review Detector

Run locally:
    streamlit run app.py

Folder structure:
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

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack


def render_html(content: str):
    """Render an HTML block via st.markdown. Strips leading whitespace from
    every line individually (not just common dedent) so that nested/indented
    HTML tags never get mistaken for a Markdown code block."""
    lines = content.strip("\n").split("\n")
    flattened = "\n".join(line.strip() for line in lines)
    st.markdown(flattened, unsafe_allow_html=True)


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

render_html(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(59, 130, 246, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(139, 92, 246, 0.12),
                transparent 30%
            ),
            #0b1120;

        color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #111827 0%,
                #0f172a 100%
            );

        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] p {
        color: #94a3b8;
    }


    /* =========================
       HERO
       ========================= */

    .hero {
        background:
            linear-gradient(
                135deg,
                #2563eb,
                #7c3aed
            );

        padding: 40px 44px;

        border-radius: 24px;

        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.35);

        margin-bottom: 30px;

        position: relative;

        overflow: hidden;
    }

    .hero::before {
        content: "";

        position: absolute;

        width: 250px;
        height: 250px;

        border-radius: 50%;

        background:
            rgba(255, 255, 255, 0.08);

        right: -80px;
        top: -100px;
    }

    .hero::after {
        content: "";

        position: absolute;

        width: 180px;
        height: 180px;

        border-radius: 50%;

        background:
            rgba(255, 255, 255, 0.05);

        left: 45%;
        bottom: -100px;
    }

    .hero-content {
        position: relative;
        z-index: 2;
    }

    .hero h1 {
        font-size: 42px;

        font-weight: 800;

        margin: 0;

        color: white;

        letter-spacing: -1px;
    }

    .hero p {
        font-size: 17px;

        color:
            rgba(255, 255, 255, 0.88);

        margin-top: 10px;

        max-width: 780px;
    }


    /* =========================
       SECTION HEADINGS
       ========================= */

    .section-title {
        font-size: 25px;

        font-weight: 750;

        color: #f8fafc;

        margin-top: 25px;

        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #94a3b8;

        font-size: 14px;

        margin-bottom: 18px;
    }


    /* =========================
       CARDS
       ========================= */

    .card {
        background:
            rgba(15, 23, 42, 0.78);

        border:
            1px solid
            rgba(148, 163, 184, 0.14);

        border-radius: 18px;

        padding: 24px;

        box-shadow:
            0 12px 30px
            rgba(0, 0, 0, 0.18);

        backdrop-filter: blur(10px);
    }


    /* =========================
       FEATURE CARDS
       ========================= */

    .feature-card {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.95),
                rgba(15, 23, 42, 0.95)
            );

        border:
            1px solid
            rgba(148, 163, 184, 0.12);

        border-radius: 18px;

        padding: 22px;

        min-height: 145px;

        transition:
            all 0.25s ease;
    }

    .feature-card:hover {
        transform:
            translateY(-5px);

        border-color:
            rgba(96, 165, 250, 0.5);

        box-shadow:
            0 15px 35px
            rgba(0, 0, 0, 0.25);
    }

    .feature-icon {
        font-size: 30px;
    }

    .feature-title {
        font-weight: 700;

        font-size: 17px;

        margin-top: 10px;

        color: #f8fafc;
    }

    .feature-text {
        font-size: 13px;

        color: #94a3b8;

        margin-top: 5px;

        line-height: 1.5;
    }


    /* =========================
       TEXT AREA
       ========================= */

    .stTextArea textarea {
        background-color:
            #0f172a !important;

        color:
            #f8fafc !important;

        border:
            1px solid
            #334155 !important;

        border-radius:
            14px !important;

        font-size:
            16px !important;

        line-height:
            1.6 !important;

        padding:
            16px !important;

        transition:
            all 0.2s ease !important;
    }

    .stTextArea textarea:focus {
        border-color:
            #60a5fa !important;

        box-shadow:
            0 0 0 2px
            rgba(96, 165, 250, 0.15) !important;
    }


    /* =========================
       SELECT BOX
       ========================= */

    div[data-baseweb="select"] > div {
        background-color:
            #0f172a !important;

        border:
            1px solid
            #334155 !important;

        border-radius:
            12px !important;

        color:
            white !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        border-radius:
            12px !important;

        border:
            1px solid
            rgba(96, 165, 250, 0.25) !important;

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #7c3aed
            ) !important;

        color:
            white !important;

        font-weight:
            700 !important;

        padding:
            10px 20px !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        transform:
            translateY(-2px) !important;

        box-shadow:
            0 8px 25px
            rgba(59, 130, 246, 0.35) !important;

        border-color:
            rgba(255, 255, 255, 0.25) !important;
    }


    /* =========================
       RESULT CARDS
       ========================= */

    .ai-result {
        background:
            linear-gradient(
                135deg,
                rgba(127, 29, 29, 0.65),
                rgba(30, 15, 20, 0.95)
            );

        border:
            1px solid
            rgba(248, 113, 113, 0.4);

        border-radius: 20px;

        padding: 28px;

        box-shadow:
            0 15px 40px
            rgba(127, 29, 29, 0.15);
    }

    .human-result {
        background:
            linear-gradient(
                135deg,
                rgba(6, 78, 59, 0.65),
                rgba(10, 30, 25, 0.95)
            );

        border:
            1px solid
            rgba(52, 211, 153, 0.4);

        border-radius: 20px;

        padding: 28px;

        box-shadow:
            0 15px 40px
            rgba(6, 78, 59, 0.15);
    }

    .result-label {
        font-size: 14px;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 1.5px;

        color: #cbd5e1;
    }

    .result-title {
        font-size: 32px;

        font-weight: 800;

        margin-top: 6px;
    }

    .result-description {
        color: #cbd5e1;

        font-size: 14px;

        margin-top: 8px;

        line-height: 1.5;
    }


    /* =========================
       STAT CARDS
       ========================= */

    .stat-card {
        background:
            #111827;

        border:
            1px solid
            rgba(148, 163, 184, 0.12);

        border-radius:
            14px;

        padding:
            16px;

        text-align:
            center;
    }

    .stat-value {
        font-size:
            23px;

        font-weight:
            800;

        color:
            #60a5fa;
    }

    .stat-label {
        font-size:
            12px;

        color:
            #94a3b8;

        margin-top:
            3px;
    }


    /* =========================
       CONFIDENCE
       ========================= */

    .confidence-box {
        background:
            rgba(15, 23, 42, 0.8);

        border:
            1px solid
            rgba(148, 163, 184, 0.12);

        border-radius:
            18px;

        padding:
            22px;

        margin-top:
            18px;
    }

    .confidence-number {
        font-size:
            38px;

        font-weight:
            850;

        color:
            #60a5fa;
    }


    /* =========================
       BADGES
       ========================= */

    .badge {
        display:
            inline-block;

        padding:
            5px 12px;

        border-radius:
            999px;

        font-size:
            12px;

        font-weight:
            700;

        background:
            rgba(96, 165, 250, 0.12);

        color:
            #93c5fd;

        border:
            1px solid
            rgba(96, 165, 250, 0.25);

        margin-right:
            5px;
    }


    /* =========================
       INFO BOX
       ========================= */

    .info-box {
        background:
            rgba(30, 41, 59, 0.65);

        border-left:
            4px solid #60a5fa;

        padding:
            15px 18px;

        border-radius:
            10px;

        color:
            #cbd5e1;

        font-size:
            14px;

        line-height:
            1.5;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align:
            center;

        color:
            #64748b;

        font-size:
            12px;

        padding-top:
            30px;

        padding-bottom:
            10px;
    }


    /* =========================
       STREAMLIT UI CLEANUP
       ========================= */

    #MainMenu {
        visibility:
            hidden;
    }

    footer {
        visibility:
            hidden;
    }

    header {
        visibility:
            hidden;
    }

    </style>
    """
)


# =========================================================
# ARTIFACT DIRECTORY
# =========================================================

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "artifacts",
)


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():

    word_vec = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "word_vectorizer.joblib",
        )
    )

    char_vec = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "char_vectorizer.joblib",
        )
    )

    scaler = joblib.load(
        os.path.join(
            ARTIFACTS_DIR,
            "style_scaler.joblib",
        )
    )

    models = {
        "Linear SVM": joblib.load(
            os.path.join(
                ARTIFACTS_DIR,
                "linear_svm.joblib",
            )
        ),

        "Logistic Regression": joblib.load(
            os.path.join(
                ARTIFACTS_DIR,
                "logistic_regression.joblib",
            )
        ),

        "XGBoost": joblib.load(
            os.path.join(
                ARTIFACTS_DIR,
                "xgboost.joblib",
            )
        ),
    }

    return (
        word_vec,
        char_vec,
        scaler,
        models,
    )


word_vec, char_vec, scaler, models = load_artifacts()


# =========================================================
# LABELS
# =========================================================

LABELS = [
    "OR (Human-written)",
    "CG (AI-generated)",
]


# =========================================================
# STYLOMETRIC FEATURES
# =========================================================

def stylometric_features(text_series):

    feats = pd.DataFrame(
        index=text_series.index
    )

    feats["char_len"] = (
        text_series.str.len()
    )

    feats["word_count"] = (
        text_series
        .str.split()
        .apply(len)
    )

    feats["avg_word_len"] = (
        feats["char_len"]
        /
        feats["word_count"].replace(
            0,
            1
        )
    )

    feats["punct_ratio"] = (
        text_series.apply(
            lambda t:
                sum(
                    1
                    for c in t
                    if c in ".,!?;:"
                )
                /
                max(len(t), 1)
        )
    )

    feats["upper_ratio"] = (
        text_series.apply(
            lambda t:
                sum(
                    1
                    for c in t
                    if c.isupper()
                )
                /
                max(len(t), 1)
        )
    )

    feats["exclaim_count"] = (
        text_series.str.count("!")
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

        conf = model.predict_proba(
            X
        )[0, 1]

    elif hasattr(model, "decision_function"):

        decision = model.decision_function(
            X
        )[0]

        conf = 1 / (
            1 + np.exp(-decision)
        )

    else:

        conf = float(pred)

    return (
        LABELS[pred],
        float(conf),
    )


# =========================================================
# SESSION STATE
# =========================================================

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "show_about" not in st.session_state:
    st.session_state.show_about = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html(
        """
        <div style="
            text-align:center;
            padding:10px 0 25px 0;
        ">

            <div style="
                font-size:50px;
            ">
                🕵️
            </div>

            <h2 style="
                margin:5px 0;
                color:#f8fafc;
            ">
                Review Detector
            </h2>

            <span class="badge">
                AI / ML PROJECT
            </span>

        </div>
        """
    )

    st.markdown(
        "### ⚡ Quick Examples"
    )

    st.caption(
        "Select an example to test the detector."
    )

    examples = {

        "🤖 AI-Generated":
            (
                "I recently purchased this item and I am "
                "extremely satisfied with the quality and "
                "performance. It exceeded my expectations "
                "and I would highly recommend it to anyone "
                "in the market for this type of product."
            ),

        "👤 Human-Written":
            (
                "took forever to arrive bc of some shipping "
                "delay but once it got here it worked fine. "
                "my cat knocked it off the counter on day 2 "
                "and it still works so thats a plus lol"
            ),

        "⚖️ Borderline":
            "Does what it says. No complaints.",
    }

    for label, text in examples.items():

        if st.button(
            label,
            use_container_width=True,
        ):

            st.session_state.review_text = text
            st.session_state.result = None

            st.rerun()

    st.divider()

    st.markdown(
        "### 🧠 Detection Models"
    )

    st.markdown(
        """
        **Linear SVM**

        Fast classification using a linear
        decision boundary.

        **Logistic Regression**

        Probability-based text classification.

        **XGBoost**

        Gradient boosting classification model.
        """
    )

    st.divider()

    st.markdown(
        "### 📊 Features"
    )

    st.markdown(
        """
        • Word-level TF-IDF  
        • Character-level TF-IDF  
        • Stylometric features  
        • Machine Learning classification
        """
    )

    st.divider()

    st.caption(
        "AI Review Detector • FYP / Portfolio Project"
    )


# =========================================================
# HERO SECTION
# =========================================================

render_html(
    """
    <div class="hero">

        <div class="hero-content">

            <div style="
                font-size:14px;
                font-weight:700;
                letter-spacing:2px;
                margin-bottom:10px;
                color:#dbeafe;
            ">
                AI / MACHINE LEARNING
            </div>

            <h1>
                🕵️ AI-Generated Review Detector
            </h1>

            <p>
                Analyze product reviews using machine learning
                and determine whether the text resembles
                human-written or AI-generated content.
            </p>

            <div style="margin-top:18px;">

                <span class="badge">
                    TF-IDF
                </span>

                <span class="badge">
                    Stylometry
                </span>

                <span class="badge">
                    ML Classification
                </span>

            </div>

        </div>

    </div>
    """
)


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                📝
            </div>

            <div class="feature-title">
                Text Analysis
            </div>

            <div class="feature-text">
                Extracts word, character and
                writing-style patterns from reviews.
            </div>

        </div>
        """
    )


with col2:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                Machine Learning
            </div>

            <div class="feature-text">
                Compare multiple trained ML models
                to classify the review.
            </div>

        </div>
        """
    )


with col3:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                📊
            </div>

            <div class="feature-title">
                Confidence Score
            </div>

            <div class="feature-text">
                View the model's estimated confidence
                for the AI-generated prediction.
            </div>

        </div>
        """
    )


# =========================================================
# ANALYSIS SECTION
# =========================================================

render_html(
    """
    <div class="section-title">
        🔍 Analyze a Review
    </div>

    <div class="section-subtitle">
        Enter a product review and select the machine
        learning model you want to use.
    </div>
    """
)


# =========================================================
# MODEL SELECTION
# =========================================================

model_col, info_col = st.columns(
    [2, 3]
)


with model_col:

    model_choice = st.selectbox(
        "🤖 Detection Model",
        list(models.keys()),
        index=0,
    )

    model = models[model_choice]


with info_col:

    descriptions = {

        "Linear SVM":
            "⚡ Fast and effective for high-dimensional TF-IDF text features.",

        "Logistic Regression":
            "📈 Provides probability-based classification.",

        "XGBoost":
            "🚀 Gradient boosting model capable of learning complex patterns.",
    }

    render_html(
        f"""
        <div class="info-box" style="
            margin-top:28px;
        ">
            {descriptions[model_choice]}
        </div>
        """
    )


# =========================================================
# REVIEW TEXT AREA
# =========================================================

review_text = st.text_area(
    "✍️ Review Text",

    value=st.session_state.review_text,

    height=190,

    placeholder=(
        "Paste a product review here...\n\n"
        "Example: The product arrived quickly and works "
        "exactly as described. The quality is excellent "
        "for the price."
    ),

    key="review_input",
)


# =========================================================
# TEXT STATISTICS
# =========================================================

word_count = len(
    review_text.split()
)

char_count = len(
    review_text
)

sentence_count = (
    review_text.count(".")
    +
    review_text.count("!")
    +
    review_text.count("?")
)


if word_count < 5:
    length_status = "SHORT"
elif word_count < 30:
    length_status = "MEDIUM"
else:
    length_status = "GOOD"


render_html(
    "<div style='height:8px'></div>"
)


stat1, stat2, stat3, stat4 = st.columns(4)


with stat1:

    render_html(
        f"""
        <div class="stat-card">

            <div class="stat-value">
                {word_count}
            </div>

            <div class="stat-label">
                WORDS
            </div>

        </div>
        """
    )


with stat2:

    render_html(
        f"""
        <div class="stat-card">

            <div class="stat-value">
                {char_count}
            </div>

            <div class="stat-label">
                CHARACTERS
            </div>

        </div>
        """
    )


with stat3:

    render_html(
        f"""
        <div class="stat-card">

            <div class="stat-value">
                {sentence_count}
            </div>

            <div class="stat-label">
                SENTENCES
            </div>

        </div>
        """
    )


with stat4:

    render_html(
        f"""
        <div class="stat-card">

            <div class="stat-value">
                {length_status}
            </div>

            <div class="stat-label">
                TEXT LENGTH
            </div>

        </div>
        """
    )


# =========================================================
# ACTION BUTTONS
# =========================================================

render_html(
    "<div style='height:18px'></div>"
)


button1, button2, button3 = st.columns(
    [2, 1, 1]
)


with button1:

    analyze = st.button(
        "🔍 Analyze Review",
        use_container_width=True,
    )


with button2:

    clear = st.button(
        "🧹 Clear",
        use_container_width=True,
    )

    if clear:

        st.session_state.review_text = ""
        st.session_state.result = None

        st.rerun()


with button3:

    about = st.button(
        "ℹ️ About",
        use_container_width=True,
    )

    if about:

        st.session_state.show_about = (
            not st.session_state.show_about
        )


# =========================================================
# ABOUT SECTION
# =========================================================

if st.session_state.show_about:

    render_html(
        """
        <div class="card">

            <h3>
                🧠 About This Project
            </h3>

            <p style="
                color:#94a3b8;
                line-height:1.7;
            ">
                This academic project uses Natural Language
                Processing and Machine Learning to classify
                reviews as either human-written or
                AI-generated.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.7;
            ">
                The system combines word-level TF-IDF,
                character-level TF-IDF and stylometric
                features before passing them to the selected
                machine learning classifier.
            </p>

        </div>
        """
    )


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not review_text.strip():

        st.warning(
            "⚠️ Please enter a review before analyzing."
        )

    elif word_count < 3:

        st.warning(
            "⚠️ This review is extremely short. "
            "The prediction may not be reliable."
        )

    else:

        with st.spinner(
            "🧠 Analyzing linguistic patterns..."
        ):

            label, conf = predict(
                review_text,
                model,
            )

        conf = float(
            np.clip(
                conf,
                0.0,
                1.0,
            )
        )

        is_ai = "CG" in label

        st.session_state.result = (
            label,
            conf,
            is_ai,
            model_choice,
        )


# =========================================================
# RESULTS
# =========================================================

if st.session_state.result is not None:

    (
        label,
        conf,
        is_ai,
        selected_model,
    ) = st.session_state.result

    st.divider()

    render_html(
        """
        <div class="section-title">
            📊 Analysis Result
        </div>

        <div class="section-subtitle">
            Classification generated by your trained
            machine learning model.
        </div>
        """
    )


    # =====================================================
    # RESULT CARD
    # =====================================================

    if is_ai:

        render_html(
            """
            <div class="ai-result">

                <div class="result-label">
                    🔴 DETECTION RESULT
                </div>

                <div class="result-title">
                    AI-Generated Review
                </div>

                <div class="result-description">
                    The review contains patterns that resemble
                    AI-generated writing.
                </div>

            </div>
            """
        )

    else:

        render_html(
            """
            <div class="human-result">

                <div class="result-label">
                    🟢 DETECTION RESULT
                </div>

                <div class="result-title">
                    Human-Written Review
                </div>

                <div class="result-description">
                    The review contains patterns that resemble
                    naturally written human text.
                </div>

            </div>
            """
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    render_html(
        f"""
        <div class="confidence-box">

            <div style="
                color:#94a3b8;
                font-size:13px;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:1px;
            ">
                AI-Generated Probability
            </div>

            <div class="confidence-number">
                {conf:.1%}
            </div>

        </div>
        """
    )


    st.progress(
        conf,
    )


    # =====================================================
    # RESULT METRICS
    # =====================================================

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Prediction",
            "AI Generated"
            if is_ai
            else
            "Human Written",
        )


    with result_col2:

        st.metric(
            "Confidence",
            f"{conf:.1%}",
        )


    with result_col3:

        st.metric(
            "Model",
            selected_model,
        )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    if conf >= 0.80:

        if is_ai:

            interpretation = (
                "🔴 High confidence — the model has a strong "
                "indication that this review is AI-generated."
            )

        else:

            interpretation = (
                "🟢 High confidence — the model strongly "
                "associates this review with human-written text."
            )

    elif conf >= 0.60:

        if is_ai:

            interpretation = (
                "🟠 Moderate confidence — the review shows "
                "some characteristics associated with "
                "AI-generated text."
            )

        else:

            interpretation = (
                "🟡 Moderate confidence — the review shows "
                "some characteristics associated with "
                "human writing."
            )

    else:

        interpretation = (
            "⚪ Low confidence — the model is relatively "
            "uncertain about this classification."
        )


    render_html(
        f"""
        <div class="info-box">
            {interpretation}
        </div>
        """
    )


    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander(
        "🔬 View Technical Analysis"
    ):

        st.markdown(
            """
            ### Features Used

            **1. Word-level TF-IDF**

            Captures important words and word combinations
            within the review.

            **2. Character-level TF-IDF**

            Captures character patterns, spelling styles,
            word fragments and writing structures.

            **3. Stylometric Features**

            The model also considers:

            - Character length
            - Word count
            - Average word length
            - Punctuation ratio
            - Uppercase ratio
            - Exclamation count

            These features are combined and passed to
            the selected machine learning classifier.
            """
        )


# =========================================================
# DISCLAIMER
# =========================================================

render_html(
    """
    <div class="card" style="
        margin-top:35px;
    ">

        <div style="
            font-size:16px;
            font-weight:700;
            margin-bottom:8px;
        ">
            ⚠️ Important Note
        </div>

        <div style="
            color:#94a3b8;
            font-size:13px;
            line-height:1.6;
        ">
            This detector is an academic machine learning
            project. Predictions are estimates rather than
            definitive proof that a review was written by AI
            or a human. Performance may vary on text outside
            the training domain.
        </div>

    </div>
    """
)


# =========================================================
# FOOTER
# =========================================================

render_html(
    """
    <div class="footer">

        🕵️ AI-Generated Review Detector

        <br>

        Machine Learning • NLP • TF-IDF • Stylometric Analysis

        <br><br>

        FYP / Academic Portfolio Project

    </div>
    """
)
