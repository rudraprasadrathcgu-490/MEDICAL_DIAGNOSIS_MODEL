import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Medical Diagnosis", layout="wide")
st.title("🩺 AI-Powered Medical Diagnosis System")
st.markdown("Upload dataset → Train models → Compare → Predict")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR)", value=True)

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # ---------------- OUTLIER REMOVAL ----------------
    if remove_outliers:
        numeric = df.select_dtypes(include=np.number)
        Q1 = numeric.quantile(0.25)
        Q3 = numeric.quantile(0.75)
        IQR = Q3 - Q1

        mask = ~((numeric < (Q1 - 1.5 * IQR)) |
                 (numeric > (Q3 + 1.5 * IQR))).any(axis=1)

        if "Disease" in df.columns and df[mask]["Disease"].nunique() >= 2:
            df = df[mask]
            st.sidebar.success("Outliers Removed")

    if "Disease" not in df.columns:
        st.error("Dataset must contain 'Disease' column.")
        st.stop()

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ---------------- MODELS ----------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(probability=True)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

        results.append([name, acc, rec, f1, cv_score])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Recall", "F1 Score", "CV Score"]
    )

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 Model Comparison")
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
        "CV Score": "{:.2%}"
    }))

    # Select Best Model
    best_model_name = results_df.sort_values(
        by="Accuracy", ascending=False
    )["Model"].values[0]

    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    st.success(f"🏆 Best Model: {best_model_name}")

    # ---------------- ROC CURVE ----------------
    if len(y.unique()) == 2:
        st.subheader("📈 ROC Curve")
        y_probs = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1])
        ax.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
        st.pyplot(fig)

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("📉 Confusion Matrix")
    y_pred = best_model.predict(X_test)
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax2)
    st.pyplot(fig2)

    # ---------------- FEATURE IMPORTANCE ----------------
    if best_model_name == "Random Forest":
        st.subheader("🔍 Top 10 Important Features")
        importances = pd.Series(
            best_model.feature_importances_,
            index=X.columns
        )
        fig3, ax3 = plt.subplots()
        importances.nlargest(10).sort_values().plot(kind="barh", ax=ax3)
        st.pyplot(fig3)

    # ---------------- DOWNLOAD MODEL ----------------
    st.subheader("⬇ Download Trained Model")
    model_data = pickle.dumps(best_model)
    st.download_button(
        "Download Model (.pkl)",
        model_data,
        file_name="medical_ai_model.pkl"
    )

else:
    st.info("Upload a dataset to begin.")
