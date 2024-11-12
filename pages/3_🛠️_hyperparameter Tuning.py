import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, precision_recall_curve

import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
from helper_functions import *  

# app design
app_meta()
set_bg_hack('Picture1.png')
html_temp = """
  <div style="background-color: rgba(0, 0, 0, 0.7); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8);
                text-align: center;">
        <h1 style="color: white; font-size: 50px; margin: 0; 
                   text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);">Model Hyperparameter Tuning 🛠️</h1>
    </div>	"""
st.markdown(html_temp, unsafe_allow_html=True)

########################################################################################3
categories = ['positive', 'negative',]

@st.cache_data
def load_and_vectorize_dataset():
    X_train, X_test, Y_train, Y_test = train_test_split(df["clean_text"],df["rating"] , train_size=0.8, random_state=123)

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
    df=''        

def train_model(n_estimators, max_depth, max_features, bootstrap):
    rf_classif = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap)
    rf_classif.fit(X_train_vec, Y_train)
    return rf_classif

## Dashboard

st.title("Random Forest :green[Experiment] :deciduous_tree: :evergreen_tree:")
st.markdown("Try different values of Random Forest classifier. Select widget values and submit model for training. Various ML metrics will be displayed after training.")

with st.form("train_model"):
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        n_estimators = st.slider("No of Estimators:", min_value=100, max_value=1000)
        max_depth = st.slider("Max Depth:", min_value=2, max_value=20)
        max_features = st.selectbox("Max Features :", options=["sqrt", "log2", None])
        bootstrap = st.checkbox("Bootstrap")
        save_model = st.checkbox("Save Model")

        submitted = st.form_submit_button("Train")

    if submitted and df is not None and not df.empty:
        rf_classif = train_model(n_estimators, max_depth, max_features, bootstrap)

        if save_model:
            dump(rf_classif, "rf_classif.dat")

        Y_test_preds = rf_classif.predict(X_test_vec)
        Y_train_preds = rf_classif.predict(X_train_vec)
        Y_test_probs = rf_classif.predict_proba(X_test_vec)

        with col2:
            col21, col22 = st.columns(2, gap="medium")
            with col21:
                st.metric("Test Accuracy", value="{:.2f} %".format(100*accuracy_score(Y_test, Y_test_preds)))
            with col22:
                st.metric("Train Accuracy", value="{:.2f} %".format(100*accuracy_score(Y_train, Y_train_preds)))

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
            fpr, tpr, _ = roc_curve(Y_test, Y_test_probs[:, 1], pos_label=rf_classif.classes_[1])
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
            precision, recall, _ = precision_recall_curve(Y_test, Y_test_probs[:, 1], pos_label=rf_classif.classes_[1])
            pr_fig, ax2 = plt.subplots(figsize=(6,6))
            ax2.plot(recall, precision, color='blue', lw=2, label="Precision-Recall curve")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Precision-Recall Curve")
            ax2.legend(loc="upper right")
            st.pyplot(pr_fig, use_container_width=True)
