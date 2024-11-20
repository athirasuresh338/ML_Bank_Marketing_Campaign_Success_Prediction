import streamlit as st
import base64


st.set_page_config(
    page_title="Conclusion",
    page_icon=":page_facing_up:",
)

def show_conclusion_page():
    st.title(":orange[Conclusion]")

    st.header("Summary of the Project")
    st.write("""
    This project focused on developing and training a machine learning model using data from a bank marketing campaign. The main objectives were to understand customer behavior, identify key features influencing customer decisions, and predict the success of marketing campaigns.
    """)
    st.header(" Relevance of this project")
    st.write("""
    This project will enable banking institutions to conduct successful marketing campaigns. Effective campaigns will encourage households to develop stronger saving habits, which in turn will enhance the banks' ability to extend credit.
    According to the latest data from the ministry of statistics, net household savings declined sharply by ₹9 trillion in the last 3 years
    The decline in household savings underscores the necessity of the project. By enabling banks to run effective marketing campaigns, the project aims to counteract this decline, encourage saving habits, and ultimately strengthen the banks' capacity to provide credit.
    """)

    st.header("Key Findings")
    st.write("""
    1. **Contact Duration**: The duration of the last contact with the customer has a substantial impact on the likelihood of a positive outcome.
    2. **Marital Status**: Married clients opted more for term deposit
    3. **Job**: Clients belonging to Management job opted more for term deposit
    4. **Age and Account balance**: Irrespective of the age and account balance of the clients most of them chose not to opt for term deposit

    """)

    st.header("Model Performance")
    st.write("""
    The machine learning model was evaluated using various performance metrics. The model showed promising results, with a high accuracy and precision in predicting customer responses. However, there is always room for improvement, and further tuning and feature engineering could enhance the model's performance.
    """)


show_conclusion_page()
