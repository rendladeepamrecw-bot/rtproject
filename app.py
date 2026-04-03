import os
import pickle
import pandas as pd
import streamlit as st


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

MODEL_FILE = "model.pkl"
DATA_FILE = "fake_job_postings.csv"

# =========================
# TRAIN MODEL (if not exists)
# =========================
def train_model():
    data = pd.read_csv(DATA_FILE)
    data = data.fillna('')

    data['text'] = (
        data['title'] + " " +
        data['description'] + " " +
        data['company_profile'] + " " +
        data['requirements'] + " " +
        data['benefits']
    )

    X = data['text']
    y = data['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])

    model.fit(X_train, y_train)

    pickle.dump(model, open(MODEL_FILE, "wb"))
    return model


# =========================
# LOAD MODEL
# =========================
if os.path.exists(MODEL_FILE):
    model = pickle.load(open(MODEL_FILE, "rb"))
else:
    model = train_model()


# =========================
# PREDICTION LOGIC
# =========================
def predict_result(company_name, email, link, stipend, image):

    score = 0
    reasons = []

    # Email check
    if email:
        if "gmail" in email or "yahoo" in email:
            score -= 1
            reasons.append("Free email domain used")
        else:
            score += 1

    # Link check
    if link:
        if "bit.ly" in link or "tinyurl" in link:
            score -= 2
            reasons.append("Shortened link detected")
        else:
            score += 1

    # Stipend
    if stipend == "Yes":
        score += 1
    elif stipend == "No":
        score -= 1

    # Image
    if image:
        score += 1

    # Convert input to text for ML
    text_input = f"""
    Company: {company_name}
    Email: {email}
    Link: {link}
    Stipend: {stipend}
    """

    prediction = model.predict([text_input])[0]

    # Final decision
    if prediction == 0 and score >= 2:
        return "Real", reasons
    elif prediction == 1 and score <= 0:
        return "Fake", reasons
    else:
        return "Suspicious", reasons


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Fake Internship Detector", page_icon="🕵️")

st.title("🕵️ Fake Internship Detection System")

company_name = st.text_input("🏢 Company Name")
company_email = st.text_input("📧 Company Email")
link = st.text_input("🔗 Internship Link")
stipend = st.radio("💰 Stipend Offered?", ["Select", "Yes", "No"])
image = st.file_uploader("🖼️ Upload Image", type=["png", "jpg", "jpeg"])

if st.button("🔍 Check Internship"):

    if not company_name or not company_email or stipend == "Select":
        st.warning("⚠️ Please fill required fields")
    else:
        result, reasons = predict_result(
            company_name,
            company_email,
            link,
            stipend,
            image
        )

        st.subheader("📊 Result")

        if result == "Real":
            st.success("✅ REAL Internship 🟢")
        elif result == "Suspicious":
            st.warning("⚠️ SUSPICIOUS Internship 🟡")
        else:
            st.error("❌ FAKE Internship 🔴")

        if reasons:
            st.write("⚠️ Reasons:")
            for r in reasons:
                st.write(f"- {r}")
