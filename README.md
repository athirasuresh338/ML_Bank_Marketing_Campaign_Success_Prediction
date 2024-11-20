# Bank Marketing Campaign Success Prediction 🚀

This repository showcases a **machine learning project** designed to predict the success of bank marketing campaigns using a **Random Forest Classifier**. The project stems from a pressing issue highlighted by the Ministry of Statistics: the decline in India's household savings rate. By leveraging machine learning, this project seeks to provide actionable insights for financial institutions to optimize their marketing strategies and encourage savings.

---

## 🌟 **Project Overview**

The goal of this project is to develop a predictive model that helps financial institutions understand key factors contributing to successful bank marketing campaigns. This can enable banks to refine their strategies, improve customer outreach, and promote financial stability.

### **Table of Contents**
1. [Background](#background)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Streamlit Web App](#streamlit-web-app)
7. [Results and Insights](#results-and-insights)
8. [Conclusion](#conclusion)
9. [Installation](#installation)

---

## 📚 **Background**
India's **household savings rate** has dropped significantly, declining from **22.7% of GDP in 2020–21** to **18.4% in 2022–23**. This trend highlights the need for innovative financial strategies to promote savings and economic resilience. By analyzing data from bank marketing campaigns, this project seeks to provide insights that financial institutions can use to improve customer engagement and campaign outcomes.

---

## 📊 **Dataset**
The dataset used in this project includes attributes such as:
- **Demographic information**: Age, job type, marital status, etc.
- **Campaign details**: Contact duration, communication type, and previous outcomes.

Source: The dataset was sourced from publicly available bank marketing campaign data.

---

## ⚙️ **Data Preprocessing**
Key preprocessing steps include:
- **Normalization**: Scaled features using `MinMaxScaler`.
- **Categorical Encoding**: Transformed categorical variables using `LabelEncoder`.
- **Balancing the Dataset**: Applied oversampling techniques (e.g., SMOTE) to address class imbalance.

---

## 🧠 **Modeling**
A **Random Forest Classifier** was chosen for its robust performance on structured data. Steps involved:
1. **Hyperparameter Tuning**: Optimized parameters for better accuracy and efficiency.
2. **Training**: Built the model using the preprocessed dataset.

---

## 📈 **Evaluation**
The model's performance was assessed using:
- **Accuracy**: Measured predictive success.
- **ROC Curve**: Visualized model performance across various thresholds.
- **Feature Importance**: Identified the most influential factors in campaign success.

---

## 🌐 **Streamlit Web App**
An interactive **Streamlit Web App** was developed for user-friendly interaction. Features include:
- User inputs for demographic and campaign details.
- Real-time predictions on campaign success likelihood.

---

## 💡 **Results and Insights**
The project uncovered several key insights:
- **Customer demographics**: Certain age groups and job types showed higher responsiveness to campaigns.
- **Campaign strategies**: Factors like contact duration and previous outcomes strongly influenced success rates.

These findings can guide banks in designing more targeted and effective campaigns.

---

## 🏁 **Conclusion**
This project demonstrates how machine learning can address economic challenges by:
- Providing actionable insights for financial institutions.
- Empowering users through an intuitive web interface.

By leveraging predictive modeling, we can encourage informed decision-making and foster financial stability.

---

### 🔧 **Installation**
To run this project locally:
1. Clone the repository:  
   ```bash
   git clone https://github.com/athirasuresh/bank-marketing-prediction.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app:  
   ```bash
   streamlit run Homepage.py
   ```

---
