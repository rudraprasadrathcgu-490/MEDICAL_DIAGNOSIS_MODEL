import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# --- Page Config ---
st.set_page_config(page_title="Medical Diagnosis AI", layout="wide")
st.title("🩺 Medical Diagnosis Prediction Dashboard")
st.markdown("Upload your medical dataset to train models and evaluate performance.")

# --- Sidebar: File Upload & Controls ---
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR)", value=True)

if uploaded_file is not None:
    # --- 1. Load & Basic Cleaning ---
    df = pd.read_csv(uploaded_file)
    df.drop_duplicates(inplace=True)
    
    # Fill missing values
    if 'Cholesterol' in df.columns:
        df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())
    
    # Outlier removal
    if remove_outliers:
        Q1 = df.quantile(numeric_only=True, q=0.25)
        Q3 = df.quantile(numeric_only=True, q=0.75)
        IQR = Q3 - Q1
        df_cleaned = df[~((df.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) |
                          (df.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)]
        if df_cleaned['Disease'].nunique() >= 2:
            df = df_cleaned
            st.sidebar.success("Outliers removed successfully.")
        else:
            st.sidebar.warning("Outlier removal skipped: would delete an entire class.")

    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    # --- 2. Training Logic ---
    if 'Disease' in df.columns:
        X = df.drop('Disease', axis=1)
        y = df['Disease']

        # Splitting (Stratified)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # Model Training (Random Forest)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # --- 3. Dashboard Metrics ---
        st.subheader("1. Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        m2.metric("Recall (Sensitivity)", f"{recall_score(y_test, y_pred):.2%}")
        m3.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")
        m4.metric("Samples", f"{len(df)}")

        # Visuals
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.write("**Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        with v_col2:
            st.write("**Feature Importance**")
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            fig2, ax2 = plt.subplots()
            importances.nlargest(10).plot(kind='barh', ax=ax2)
            st.pyplot(fig2)

        # --- 4. Interactive Prediction Tool ---
        st.divider()
        st.subheader("2. Patient Diagnosis Tool")
        st.markdown("Enter patient details below. Binary fields (like Gender or Smoking) use dropdowns.")
        
        input_data = {}
        # Create a grid for inputs
        grid_cols = st.columns(4)
        
        for i, col_name in enumerate(X.columns):
            with grid_cols[i % 4]:
                # Check if it's a binary/dummy column
                if X[col_name].dropna().isin([0, 1]).all():
                    # Clean up the label for the UI
                    clean_label = col_name.replace('_Yes', '').replace('_', ' ')
                    choice = st.selectbox(clean_label, options=["No", "Yes"], index=0, key=col_name)
                    input_data[col_name] = 1 if choice == "Yes" else 0
                else:
                    input_data[col_name] = st.number_input(
                        col_name.replace('_', ' '), 
                        value=float(X[col_name].mean()),
                        key=col_name
                    )

        if st.button("Analyze Patient Data", type="primary"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            prediction = rf.predict(input_scaled)[0]
            probs = rf.predict_proba(input_scaled)[0]
            confidence = probs[1] if prediction == 1 else probs[0]

            if prediction == 1:
                st.error(f"### 🚨 Result: High Risk (Confidence: {confidence:.2%})")
            else:
                st.success(f"### ✅ Result: Low Risk (Confidence: {confidence:.2%})")
                
    else:
        st.error("Dataset must contain a 'Disease' column for target labels.")
else:
    st.info("👋 Welcome! Please upload your medical CSV file in the sidebar to start the analysis.")