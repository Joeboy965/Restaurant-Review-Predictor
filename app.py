import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="ReviewRate | Rating Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_models():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
except Exception as e:
    st.error(
        "The model files could not be loaded. "
        "Ensure that random_forest_model.pkl and "
        "tfidf_vectorizer.pkl are in the same folder as app.py."
    )
    st.stop()


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #101827;
        color: #F5F5F5;
    }

    [data-testid="stSidebar"] {
        background-color: #172235;
    }

    .hero {
        padding: 30px 10px 20px 10px;
    }

    .brand {
        color: #FF9F43;
        font-size: 17px;
        font-weight: bold;
        letter-spacing: 2px;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 43px;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 12px;
    }

    .hero-text {
        color: #B8C4D8;
        font-size: 17px;
        line-height: 1.7;
        max-width: 750px;
    }

    .section-title {
        color: #FFFFFF;
        font-size: 23px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .info-card {
        background-color: #1B2940;
        border: 1px solid #30415C;
        padding: 22px;
        border-radius: 16px;
        margin-bottom: 15px;
    }

    .result-card {
        background: linear-gradient(135deg, #253653, #19263B);
        border: 1px solid #FF9F43;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
    }

    .rating-number {
        color: #FFB454;
        font-size: 55px;
        font-weight: 800;
    }

    .rating-stars {
        color: #FFB454;
        font-size: 28px;
        letter-spacing: 4px;
    }

    .small-text {
        color: #B8C4D8;
        font-size: 14px;
    }

    div.stButton > button {
        background-color: #FF9F43;
        color: #101827;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 700;
        width: 100%;
        min-height: 48px;
    }

    div.stButton > button:hover {
        background-color: #FFB454;
        color: #101827;
        border: none;
    }

    .footer {
        text-align: center;
        color: #8D9BB0;
        font-size: 13px;
        padding: 35px 0 15px 0;
    }

    [data-testid="stMetric"] {
        background-color: #1B2940;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30415C;
    }

    .stTextArea textarea {
        background-color: #F7F9FC;
        color: #172235;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ⭐ ReviewRate")
    st.caption("Restaurant Review Rating Predictor")

    st.markdown("---")

    st.markdown("### About the project")

    st.write(
        "ReviewRate uses a machine learning model to estimate "
        "a restaurant review's rating based on the review text "
        "and selected review details."
    )

    st.markdown("---")

    st.markdown("### How it works")

    st.write("**01.** Enter a restaurant review.")
    st.write("**02.** Add the review details.")
    st.write("**03.** Get the predicted rating.")

    st.markdown("---")

    st.caption("Machine Learning Project")
    st.caption("Model: Random Forest Regressor")


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="brand">RESTAURANT INTELLIGENCE</div>
    <div class="hero-title">
        Understand the experience.<br>
        Predict the rating.
    </div>
    <p class="hero-text">
        Enter a restaurant review and let our machine learning
        model estimate the rating based on the review's content
        and details.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ---------------- INPUT SECTION ----------------

left_column, right_column = st.columns(
    [1.5, 1],
    gap="large"
)

with left_column:

    st.markdown(
        '<div class="section-title">📝 Enter a Review</div>',
        unsafe_allow_html=True
    )

    with st.form("review_form"):

        review = st.text_area(
            "Restaurant review",
            placeholder=(
                "Example: The food was delicious, "
                "the staff were friendly, and the service "
                "was excellent. I would gladly visit again."
            ),
            height=200,
            help="Enter the review you want the model to analyse."
        )

        st.markdown("#### Additional Details")

        pictures = st.number_input(
            "Number of pictures attached",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )

        review_date = st.date_input(
            "Review date",
            value=datetime.now().date()
        )

        review_time = st.time_input(
            "Review time",
            value=datetime.now().time().replace(
                second=0,
                microsecond=0
            )
        )

        submitted = st.form_submit_button(
            "⭐ Predict Restaurant Rating"
        )


with right_column:

    st.markdown(
        '<div class="section-title">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>🤖 Machine Learning</h3>
        <p style="color:#B8C4D8;">
            A trained Random Forest model analyses review
            information to estimate a rating.
        </p>
    </div>

    <div class="info-card">
        <h3>💬 Review Analysis</h3>
        <p style="color:#B8C4D8;">
            TF-IDF converts the review text into numerical
            features that the model can process.
        </p>
    </div>

    <div class="info-card">
        <h3>⚡ Instant Prediction</h3>
        <p style="color:#B8C4D8;">
            Submit a review to see the rating predicted
            by the trained model.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- PREDICTION ----------------

if submitted:

    if not review.strip():

        st.warning("Please enter a restaurant review first.")

    else:

        with st.spinner("Analysing your review..."):

            # Convert review text into TF-IDF features
            review_features = vectorizer.transform([review])

            # Extract numerical features
            review_length = len(review)
            word_count = len(review.split())
            review_hour = review_time.hour
            review_day = review_date.day
            review_month = review_date.month

            numeric_features = np.array([[
                review_length,
                word_count,
                pictures,
                review_hour,
                review_day,
                review_month
            ]], dtype=float)

            # Combine text and numerical features
            numeric_sparse = csr_matrix(numeric_features)

            combined_features = hstack([
                review_features,
                numeric_sparse
            ]).tocsr()

            # Predict rating
            prediction = float(
                np.asarray(
                    model.predict(combined_features)
                ).ravel()[0]
            )

        # Display results
        st.markdown("---")

        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )

        result_column, details_column = st.columns(
            [1, 1],
            gap="large"
        )

        with result_column:

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="color:#B8C4D8;font-size:16px;">
                        PREDICTED RESTAURANT RATING
                    </div>
                    <div class="rating-number">
                        {prediction:.2f}<span style="font-size:22px;">
                        / 5</span>
                    </div>
                    <div class="rating-stars">
                        {"★" * max(0, min(5, round(prediction)))}
                        {"☆" * (5 - max(0, min(5, round(prediction))))}
                    </div>
                    <p style="color:#B8C4D8;">
                        Estimated by the Random Forest model
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with details_column:

            st.markdown("#### Review Details")

            st.metric("Characters", f"{review_length:,}")
            st.metric("Words", f"{word_count:,}")
            st.metric("Pictures", int(pictures))

            st.progress(
                max(0.0, min(1.0, prediction / 5.0))
            )

            st.caption(
                "The prediction is an estimate, not a guarantee "
                "of the restaurant's actual rating."
            )

        st.markdown("#### Review submitted")

        st.info(review)

        st.markdown(
            '<p class="small-text">Note: Prediction quality depends '
            'on the training data and the features used to train '
            'the model.</p>',
            unsafe_allow_html=True
        )


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    ReviewRate · Restaurant Review Rating Predictor<br>
    Powered by Machine Learning
</div>
""", unsafe_allow_html=True)