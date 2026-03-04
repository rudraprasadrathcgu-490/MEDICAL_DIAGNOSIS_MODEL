import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Health Risk System", layout="wide")
st.title("🩺 AI Health Risk Prediction System")
st.markdown("Upload dataset → Train model → Enter patient details → Get health report")

# ---------------- DATASET UPLOAD ----------------
uploaded_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file).drop_duplicates()

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

    if "Disease" not in df.columns:
        st.error("Dataset must contain a column named 'Disease'.")
        st.stop()

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    if y.nunique() < 2:
        st.error("Target must contain at least 2 classes.")
        st.stop()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model trained successfully ✅ | Accuracy: {acc:.2%}")

    st.divider()
    st.subheader("🧪 Enter Patient Details")

    input_data = {}
    cols = st.columns(3)

    gender_columns = [col for col in feature_columns if "Gender_" in col]
    gender_selected = None

    if len(gender_columns) > 0:
        gender_selected = st.selectbox("Gender", ["Male", "Female", "Transgender"])

    for i, feature in enumerate(feature_columns):

        with cols[i % 3]:

            if feature in gender_columns:
                continue

            elif X[feature].dropna().isin([0, 1]).all():
                val = st.selectbox(feature.replace("_", " "), ["No", "Yes"], key=feature)
                input_data[feature] = 1 if val == "Yes" else 0

            else:
                input_data[feature] = st.number_input(
                    feature.replace("_", " "),
                    value=float(X[feature].mean()),
                    key=feature
                )

    if gender_selected:
        for col in gender_columns:
            input_data[col] = 0
        selected_col = f"Gender_{gender_selected}"
        if selected_col in gender_columns:
            input_data[selected_col] = 1

    # ---------------- ANALYZE BUTTON ----------------
    if st.button("Analyze Health Condition", type="primary"):

        input_df = pd.DataFrame([input_data])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        confidence = np.max(probs)

        if prediction == 0:
            health_score = int(confidence * 100)
        else:
            health_score = int((1 - confidence) * 100)

        if health_score >= 85:
            risk_label_ui = "🟢 Best Condition"
            risk_label_pdf = "Best Condition"
        elif health_score >= 65:
            risk_label_ui = "🟡 Average Condition"
            risk_label_pdf = "Average Condition"
        elif health_score >= 40:
            risk_label_ui = "🟠 Mild Risk"
            risk_label_pdf = "Mild Risk"
        else:
            risk_label_ui = "🔴 High Risk"
            risk_label_pdf = "High Risk"

        st.subheader("📊 Health Assessment Result")
        st.write(f"### {risk_label_ui}")
        st.write(f"### Health Score: {health_score}/100")
        st.write(f"Confidence: {confidence:.2%}")

        # ---------------- RISK GAUGE ----------------
        st.subheader("📈 Risk Meter")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([health_score, 100 - health_score], startangle=90)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.text(0, 0, f"{health_score}", ha='center', va='center', fontsize=22)
        ax.set_aspect('equal')
        st.pyplot(fig)

        # ---------------- RECOMMENDATIONS ----------------
        st.subheader("💡 Health Recommendations")

        if health_score >= 85:
            recommendation = "Maintain your current healthy lifestyle. Continue balanced diet and regular exercise."
        elif health_score >= 65:
            recommendation = "Improve diet, increase physical activity, and monitor health metrics regularly."
        elif health_score >= 40:
            recommendation = "Consult a doctor. Manage cholesterol, blood pressure, sugar levels, and avoid smoking."
        else:
            recommendation = "Immediate medical consultation recommended. Lifestyle modification required urgently."

        st.info(recommendation)

        # ---------------- PDF REPORT WITH CHART ----------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="AI Health Risk Assessment Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Risk Category: {risk_label_pdf}", ln=True)
        pdf.cell(200, 10, txt=f"Health Score: {health_score}/100", ln=True)
        pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=True)
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"Recommendation: {recommendation}")
        pdf.ln(10)

        # Save chart temporarily
        chart_path = "risk_meter.png"
        fig.savefig(chart_path, bbox_inches="tight")

        # Insert chart into PDF
        pdf.image(chart_path, x=30, w=150)

        # Generate PDF correctly
        pdf_bytes = pdf.output(dest='S').encode('latin-1')

        st.download_button(
            label="⬇ Download Health Report (PDF)",
            data=pdf_bytes,
            file_name="health_report.pdf",
            mime="application/pdf"
        )

        # Remove temporary image
        if os.path.exists(chart_path):
            os.remove(chart_path)

else:
    st.info("Upload dataset to begin.")
