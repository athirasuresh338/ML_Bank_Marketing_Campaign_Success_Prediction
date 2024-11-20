import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

st.set_page_config(
    page_title="About the Dataset",
    page_icon=":bookmark_tabs:",
)

def load_data(path):
    data = pd.read_csv(path)
    return data

df = load_data('bank-full.csv')


st.title(":orange[Bank Marketing Campaign Dataset]")

st.write("""
            #### The data is related with direct marketing campaigns of a Portuguese banking institution. 
            The marketing campaigns were based on phone calls. 
            Often, more than one contact to the same client was required, in order to access if the product (Bank term deposit) would be ('Yes') or not ('No') subscribed.
            
            #### Here is the dataset used for training the ML model 	:chart_with_upwards_trend: :
         """)
st.write("- [Link to Bank Marketing Campaign Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)")
st.dataframe(df)

st.subheader("Dataset Information")

p=st.selectbox('Select the information you want to know',['Size','Shape','Describe'])
if p=='Size':
    dataset_size = df.size
    st.write(f"**Size**: {dataset_size}")
elif p=='Shape':
    dataset_shape = df.shape
    st.write(f"**Shape**: {dataset_shape}")
elif p=='Describe':
    st.dataframe(df.describe())


st.subheader("Visualizations of Dataset")

# Select columns to visualize
numeric_columns = df.select_dtypes(['float64', 'int64']).columns
categorical_columns = df.select_dtypes(['object']).columns


# Joint plot
st.write("### Joint Plot")
on=st.toggle("Show Joint Plot")
if on:
    selected_joint_x = st.selectbox("Select X-axis variable for joint plot", numeric_columns, key="joint_x")
    selected_joint_y = st.selectbox("Select Y-axis variable for joint plot", numeric_columns, key="joint_y")
    selected_joint_hue = st.selectbox("Select hue for joint plot", categorical_columns, key="joint_hue")
    fig = sns.jointplot(x=selected_joint_x, y=selected_joint_y, data=df, hue=selected_joint_hue, kind="scatter")
    st.pyplot(fig)

# Count plot
st.write("### Count Plot")
on1=st.toggle("Show Count Plot")
if on1:
    selected_count_column = st.selectbox("Select column for count plot", categorical_columns, key="count")
    selected_count_hue = st.selectbox("Select hue for count plot", categorical_columns, key="count_hue")
    fig, ax = plt.subplots()
    sns.countplot(x=selected_count_column, hue=selected_count_hue, data=df, ax=ax)
    plt.xticks(rotation=90)  # Rotate x-axis labels if necessary
    st.pyplot(fig)

# Correlation heatmap
st.write("### Correlation Heatmap")
on2=st.toggle("Show Correlation Heatmap")
if on2:
    df1= load_data('bank_cleaned.csv')
    corr_matrix = df1.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix,annot=True, fmt=".2f", annot_kws={"size": 5},linewidths=.5,cmap='coolwarm', ax=ax,)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)