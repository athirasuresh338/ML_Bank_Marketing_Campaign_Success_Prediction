import pickle
import streamlit as st
import base64
from joblib import load


st.set_page_config(
    page_title="Model Prediction",
    page_icon=":robot_face:",
)

def main():
    st.title(':orange[PREDICTING THE SUCCESS OF BANK MARKETING EFFORTS]')

    # model=pickle.load(open('rf_model_1.sav','rb'))
    model = load('rf_model.joblib')
    # scaler=pickle.load(open('scaler_1.sav','rb'))
    scaler = load('scaler.joblib')

    age=st.text_input('Enter the age of client')

    job=st.selectbox('Enter the job of the client',['Blue-collar','Management','Technician','Admin','Services','Retired','Self-employed','Entrepreneur','Unemployed','Housemaid','Student'])
    if job=='Blue-collar':
        j=1
    elif job=='Management':
        j=4
    elif job=='Technician':
        j=9
    elif job=='Admin':
        j=0
    elif job=='Services':
        j=7
    elif job=='Retired':
        j=5
    elif job=='Self-employed':
        j=6
    elif job=='Entrepreneur':
        j=2
    elif job=='Unemployed':
        j=10
    elif job=='Housemaid':
        j=3
    else:
        j=8

    marital_status=st.radio('Enter the marital status of client',['Single','Married','Divorced'])
    if marital_status=='Single':
        ms=2
    elif marital_status=='Married':
        ms=1
    else:
        ms=0

    educational_status=st.radio('Enter the educational status of client', ['Primary', 'Secondary', 'Tertiary'])
    if educational_status=='Primary':
        es=0
    elif educational_status=='Secondary':
        es=1
    else:
        es=2

    loan_default_status=st.radio('Loan default history of client',['No','Yes'])
    if loan_default_status=='No':
        lds=0
    else:
        lds=1

    account_balance=st.text_input('Enter the account balance of client')

    housing_status = st.radio('Does the client owns a house', ['No', 'Yes'])
    if housing_status=='No':
        hs=0
    else:
        hs=1

    loan_status=st.radio('Does the client has taken any loan', ['No', 'Yes'])
    if loan_status=='No':
        ls=0
    else:
        ls=1

    contact_mode=st.radio('Select the contact mode of client', ['Cellular', 'Telephone'])
    if contact_mode=='Cellular':
        cm=0
    else:
        cm=1

    contacted_day_of_month = st.slider('Enter the day of the month when the client was coontacted', min_value=1, max_value=31)

    month=st.selectbox('Enter the month in which client was contacted',['January','Febuary','March','April','May','June','July','August','September','October','November','December'])
    if month=='January':
        m=4
    elif month=='Febuary':
        m=3
    elif month=='March':
        m=7
    elif month=='April':
        m=0
    elif month=='May':
        m=8
    elif month=='June':
        m=6
    elif month=='July':
        m=5
    elif month=='August':
        m=1
    elif month=='September':
        m=11
    elif month=='October':
        m=10
    elif month=='November':
        m=9
    else:
        m=2

    duration_of_call=st.text_input('Enter the duration of call with the client in second')

    campaign_call = st.text_input('Enter the number of campaign calls made to the client')

    no_days_passed = st.text_input('Enter the number of days passed after the last campaign calls made to the client')

    previous_campaign_call = st.text_input('Enter the number of contacts made to the client before this campaign')

    pred=st.button(':orange[PREDICT]')
    if pred:
        try:
           prediction=model.predict(scaler.transform([[age,j,ms,es,lds,account_balance,hs,ls,cm,contacted_day_of_month,m,duration_of_call,campaign_call,no_days_passed,previous_campaign_call]]))
           if prediction==0:
              st.write('#### Client will not subscribe to Term deposit :x:')
           else:
              st.write('#### Client will subscribe to Term deposit :white_check_mark:')
        except:
            st.write('#### Make sure you have given all the necessary inputs :exclamation:')

main()




