import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc,roc_curve
from imblearn.over_sampling import SMOTE
import base64

st.set_page_config(
    page_title="About the ML Model",
    page_icon=":robot_face:",
)

def show_about_model_page():
    st.title(":orange[About the Model]")

    st.header("Model Overview")
    st.write("""
    This machine learning model is  :orange[**Random Forest Classifier**]. It is trained on data from a bank marketing campaign, with the aim of predicting the success of marketing efforts.
    """)


    st.header("Data Preprocessing")
    st.write("""
        The initial step in training the model involved preprocessing the data. This included:
        - Handling missing values
        - Encoding categorical variables
        - Scaling numerical features
        """)
    st.write("""
        These steps ensured that the data was clean and ready for model training.
        """)

    st.header("Handling Imbalanced Data")
    st.write("""
        The dataset exhibited class imbalance, which could negatively impact the model's performance on the minority class. To address this, the data was oversampled using techniques SMOTE (Synthetic Minority Over-sampling Technique). This helped in creating a balanced dataset for training.
        """)

    st.header("Model Selection")
    st.write("""
    The following models were tested:
    - K-Nearest Neighbors (KNN)
    - Support Vector Classifier (SVC)
    - Naive Bayes (NB)
    - Decision Tree (DT)
    - Random Forest (RF)
    - AdaBoost
    - Gradient Boosting
    - XGBoost""")

    st.subheader("Model Perfromance Comparison")
    acd=pd.read_csv('acd.csv')
    st.dataframe(acd)

    st.write("""
        A **Random Forest Classifier** was chosen as the model for this project as it gave high accuracy,precession and recall compared to other models. Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of the prediction.
        """)


    st.header("Hyperparameter Tuning")
    st.write("""
        Hyperparameter tuning was performed to optimize the model's performance. Various combinations of parameters were tested using grid search and cross-validation. The key hyperparameters tuned included:
        - Criterion (Gini or Entropy)
        - Maximum depth of the trees
        - Random state
        """)

    st.write(""" The best combination of hyperparameters was selected based on the performance on the validation set.
        """)

    st.header("Training the Model")
    st.write("""
        The model was trained on both the original and oversampled datasets. The training process involved fitting the model to the training data and evaluating its performance using metrics such as accuracy, precision, and recall.
        """)

    st.header("Evaluation and Validation")
    st.write("""
        After training the model, it was evaluated on the test dataset to assess its performance. The evaluation metrics provided insights into how well the model generalizes to unseen data. The confusion matrix, accuracy, precision, and recall were key metrics used in this evaluation.
        """)

    df = pd.read_csv('bank-full.csv')
    df.drop(['Outcome of previous campaign'], axis=1, inplace=True)
    attributes1 = ['Job', 'Education', 'Contact details']
    for attribute in attributes1:
        df[attribute] = df[attribute].replace('unknown', np.nan)
    attributes2 = ['Job', 'Education', 'Contact details']
    for attribute in attributes2:
        df[attribute] = df[attribute].fillna(df[attribute].mode()[0])
    attributes3 = ['Job', 'Marital status', 'Education', 'Loan default', 'Housing', 'Loan', 'Contact details', 'Month of campaign', 'Outcome of present campaign']
    le = LabelEncoder()
    for attribute in attributes3:
        df[attribute] = le.fit_transform(df[attribute])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = MinMaxScaler()
    os = SMOTE(random_state=1)
    X_os, y_os = os.fit_resample(X, y)
    X_os_scaled = scaler.fit_transform(X_os)
    X_os_train, X_os_test, y_os_train, y_os_test = train_test_split(X_os_scaled, y_os, test_size=0.3, random_state=1)
    rf=RandomForestClassifier(criterion='gini',max_depth=20,random_state=1)
    rf.fit(X_os_train, y_os_train)
    y_os_pred = rf.predict(X_os_test)
    acs=accuracy_score(y_os_test, y_os_pred)
    report=classification_report(y_os_test, y_os_pred, output_dict=True)
    cm = confusion_matrix(y_os_test, y_os_pred)
    report_df = pd.DataFrame(report).transpose()

    st.header(':orange[RandomForestClassifier]')

    st.subheader(":round_pushpin: Accuracy score : 92 %")
    # st.text(f'{acs * 100:.2f} %')

    st.subheader(":round_pushpin: Classification Report")
    st.dataframe(pd.read_csv('clf_report_df.csv'))

    st.subheader(":round_pushpin: Confusion Matrix")

    on=st.toggle('Show Confusion Matrix')
    if on:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot(plt)

    st.subheader(":round_pushpin: AUC - ROC Curve")

    on1=st.toggle('Show AUC - ROC Curve')
    if on1:
        y_pred_prob=rf.predict_proba(X_os_test)[:,-1]
        fpr, tpr, thresholds = roc_curve(y_os_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area ={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)



show_about_model_page()
