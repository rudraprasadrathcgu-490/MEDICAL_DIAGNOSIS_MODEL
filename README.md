# 🩺 AI-Powered Medical Diagnosis System

An end-to-end Machine Learning web application built using **Streamlit** that allows users to upload medical datasets, train multiple ML models, compare performance, visualize results, and download the best trained model.

---

## 🚀 Live Features

- 📂 Upload any medical dataset (CSV format)
- 🧹 Automatic data cleaning & duplicate removal
- 📊 Optional IQR-based outlier removal
- 🔢 Automatic categorical encoding
- 🤖 Multiple ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- 📈 Model comparison dashboard
- 🔁 5-Fold Cross Validation
- 📉 Confusion Matrix visualization
- 📈 ROC Curve (Binary classification)
- 🔍 Feature Importance (Random Forest)
- ⬇ Download trained model (.pkl)

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

---

## 📊 How It Works

1. Upload a medical dataset containing a target column named `Disease`
2. App automatically:
   - Cleans missing values
   - Removes duplicates
   - Performs encoding
   - Scales features
3. Trains 3 ML models
4. Compares performance using:
   - Accuracy
   - Recall
   - F1 Score
   - Cross-validation score
5. Selects the best-performing model
6. Displays:
   - Confusion Matrix
   - ROC Curve (if binary)
   - Feature Importance
7. Allows model download

---

## ALWAYS READY TO LEARN 
