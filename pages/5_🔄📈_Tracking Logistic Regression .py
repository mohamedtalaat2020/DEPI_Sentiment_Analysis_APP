#import mlflow.keras
import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, precision_recall_curve
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt


# Set MLflow to use the default local tracking server
mlflow.set_tracking_uri("https://dagshub.com/mohamedtalaat2020/DEPI_Sentiment_Analysis_APP.mlflow")  # This creates an 'mlruns' directory in the current working directory

# Create or set the experiment
mlflow.set_experiment("Logistic_Regression_Experiments")

import os
os.environ['MLFLOW_TRACKING_USERNAME'] = 'mohamedtalaat2020' # 'learnpythonlanguage'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '282150b752f528876ab0c5122f0e0193b476275d' #
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/mohamedtalaat2020/DEPI_Sentiment_Analysis_APP.mlflow' 


# MLflow setup
#mlflow.set_tracking_uri("http://localhost:5000")  # Use your MLflow tracking server URI if not local
#mlflow.set_experiment("Logistic_Regression_Tuning")

# App design
st.markdown("""
    <div style="background-color: rgba(0, 0, 0, 0.7); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8);
                text-align: center;">
        <h1 style="color: white; font-size: 50px; margin: 0; 
                   text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);">Model Hyperparameter Tuning 🛠️</h1>
    </div>	
""", unsafe_allow_html=True)

# Load and preprocess data
categories = ['positive', 'negative']

@st.cache_data
def load_and_vectorize_dataset():
    X_train, X_test, Y_train, Y_test = train_test_split(df["clean_text"], df["rating"], train_size=0.8, random_state=123)
    vectorizer = CountVectorizer(max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    return X_train_vec, X_test_vec, Y_train, Y_test

if 'uploaded_data' in st.session_state:
    df = st.session_state['uploaded_data']
    X_train_vec, X_test_vec, Y_train, Y_test = load_and_vectorize_dataset()
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
else:
    df = ''

# Define the training function with MLflow logging
def train_model(C, max_iter, solver, penalty, class_weight, tol):
    with mlflow.start_run():
        log_reg = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            penalty=penalty,
            class_weight=class_weight,
            tol=tol,
            random_state=123
        )
        log_reg.fit(X_train_vec, Y_train)
        
        # Log parameters
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("tol", tol)

        # Log metrics
        Y_test_preds = log_reg.predict(X_test_vec)
        test_accuracy = accuracy_score(Y_test, Y_test_preds)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        Y_train_preds = log_reg.predict(X_train_vec)
        train_accuracy = accuracy_score(Y_train, Y_train_preds)
        mlflow.log_metric("train_accuracy", train_accuracy)
        
        return log_reg, test_accuracy, train_accuracy

# Dashboard
st.title("Logistic Regression :green[Experiment] ⚖️ with MLflow Tracking")
st.markdown("Try different values of Logistic Regression classifier. Select widget values and submit model for training. Various ML metrics will be displayed after training.")

with st.form("train_model"):
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        C = st.slider("Inverse of Regularization Strength (C):", min_value=0.01, max_value=10.0, step=0.1)
        max_iter = st.slider("Max Iterations:", min_value=50, max_value=500, step=50)
        tol = st.number_input("Tolerance (tol)", min_value=1e-5, max_value=1e-1, value=1e-4, format="%.5f")
        
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
        
        class_weight_option = st.selectbox("Class Weight", ["None", "balanced"])
        class_weight = None if class_weight_option == "None" else "balanced"
        
        save_model = st.checkbox("Save Model")

        submitted = st.form_submit_button("Train")

    if submitted and df is not None and not df.empty:
        log_reg, test_accuracy, train_accuracy = train_model(C, max_iter, solver, penalty, class_weight, tol)

        # Save the model if checkbox is selected
        if save_model:
            dump(log_reg, "log_reg_model.dat")
            mlflow.sklearn.log_model(log_reg, "log_reg_model")

        Y_test_preds = log_reg.predict(X_test_vec)
        Y_test_probs = log_reg.predict_proba(X_test_vec)

        with col2:
            col21, col22 = st.columns(2, gap="medium")
            with col21:
                st.metric("Test Accuracy", value="{:.2f} %".format(100 * test_accuracy))
            with col22:
                st.metric("Train Accuracy", value="{:.2f} %".format(100 * train_accuracy))

            st.markdown("### Confusion Matrix")
            conf_mat = confusion_matrix(Y_test, Y_test_preds)
            conf_mat_fig, ax = plt.subplots(figsize=(6,6))
            cax = ax.matshow(conf_mat, cmap="Blues")
            plt.colorbar(cax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            for (i, j), val in np.ndenumerate(conf_mat):
                ax.text(j, i, f'{val}', ha='center', va='center', color='red')
            st.pyplot(conf_mat_fig, use_container_width=True)

        st.markdown("### Classification Report:")
        st.code("==" + classification_report(Y_test, Y_test_preds, target_names=categories))

        st.markdown("### ROC & Precision-Recall Curves")
        col1, col2 = st.columns(2, gap="medium")

        # ROC Curve
        with col1:
            fpr, tpr, _ = roc_curve(Y_test, Y_test_probs[:, 1], pos_label=log_reg.classes_[1])
            roc_fig, ax1 = plt.subplots(figsize=(6,6))
            ax1.plot(fpr, tpr, color='blue', lw=2, label="ROC curve")
            ax1.plot([0, 1], [0, 1], 'k--', lw=2)
            ax1.set_xlabel("False Positive Rate")
            ax1.set_ylabel("True Positive Rate")
            ax1.set_title("ROC Curve")
            ax1.legend(loc="lower right")
            st.pyplot(roc_fig, use_container_width=True)
        
        # Precision-Recall Curve
        with col2:
            precision, recall, _ = precision_recall_curve(Y_test, Y_test_probs[:, 1], pos_label=log_reg.classes_[1])
            pr_fig, ax2 = plt.subplots(figsize=(6,6))
            ax2.plot(recall, precision, color='blue', lw=2, label="Precision-Recall curve")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Precision-Recall Curve")
            ax2.legend(loc="upper right")
            st.pyplot(pr_fig, use_container_width=True)
