import streamlit as st
import base64

st.set_page_config(
    page_title="Bank Marketing",
    page_icon=":bank:",
)

st.title("Machine Learning Project")

st.write("# Introduction")

st.write("""
    Streamlit web app created for the purpose of my Machine Learning (ML) project. 
    The project involves creating and training an ML model on data from a bank marketing campaign.

    ## Purpose of the Project
    - To demonstrate the application of Machine Learning techniques on real-world data.
    - To help banking institutions analyze the effectiveness of their campaigns and better target their future campaigns
    - To predict the likelihood of a customer subscribing to a term deposit based on the marketing campaign data.
    - To understand which factor of their campaign influenced their clients more
    - Optimize campaign resource allocation
    - Data-Driven Decision Making
    - Thereby increasing the efficiency of their campaign

    ## Features of the App
    - User-friendly interface.
    - Interactive data visualizations to explore the dataset.
    - Model predictions based on user inputs.
    
    ## Getting Started
    Use the sidebar to navigate through the different sections of the app. Each section provides specific functionalities
    and visualizations related to the project's objectives.

    Enjoy exploring the app!
""")

st.write("Created by :blue[Athira Suresh]")





#streamlit run Homepage.py