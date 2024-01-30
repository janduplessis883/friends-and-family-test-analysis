import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime
from datetime import date
from matplotlib.patches import Patch
import time  
from openai import OpenAI
from streamlit_extras.buy_me_a_coffee import button

client = OpenAI()

from utils import *

st.set_page_config(page_title="AI MedReview: FFT", layout="wide")

html = """
<style>
.gradient-text {
    background: linear-gradient(45deg, #e16d33, #ae4f4d, #f3de82, #d59c0d);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 3em;
    font-weight: bold;
}

</style>
<div class="gradient-text">AI MedReview: FFT</div>

"""
# Render the HTML in the Streamlit app
st.markdown(html, unsafe_allow_html=True)

def load_data():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df

data = load_data()

def load_timedata():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    df.set_index("time", inplace=True)
    return df

# Calculate monthly averages
data_time = load_timedata()

monthly_avg = data_time["rating_score"].resample("M").mean()
monthly_avg_df = monthly_avg.reset_index()
monthly_avg_df.columns = ["Month", "Average Rating"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Dashboard", "Feedback Classification", "Improvement Suggestions", "Rating & Sentiment Analysis", "GPT-4 Feedback Summary", "Word Cloud", "Dataframe", "About"])

with tab1:
   st.subheader("Dashboard")

col1, col2 = st.columns(2)

with col1.container(height=1000):
    temperature = "-10"
    st.markdown(":rainbow[**Feedback**]")
    st.write("blue, green, orange, red, violet, gray/grey, and rainbow")
    st.write(f"temprature: :red[{temperature}]")
    st.markdown(
        """:blue[**Feedback**] in bold textThis command forces pip to reinstall the package, which can sometimes resolve path issues.
:orange[Improvement Suggestion:] Ensure that the Python interpreter you're using to run your script is the one where hand883 is installed. Sometimes, especially in systems where multiple Python environments are present, it's easy to install a package in one interpreter and inadvertently use a different interpreter to run your script.
Importing Submodules: If your package has submodules, make sure you're importing them correctly. For instance, if hand883 has a submodule named submodule, you might need to import it explicitly:This is a new heading** in bold textThis command forces pip to reinstall the package, which can sometimes resolve path issues.
Python Path: Ensure that the Python interpreter you're using to run your script is the one where hand883 is installed. Sometimes, especially in systems where multiple Python environments are present, it's easy to install a package in one interpreter and inadvertently use a different interpreter to run your script.
Importing Submodules: If your package has submodules, make sure you're importing them correctly. For instance, if hand883 has a submodule named submodule, you might need to import it explicitly:"""
    )

with col2.container(height=300):
    st.write("Column 2")
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a series of x values
    x = np.linspace(0, 10, 100)

    # Generate corresponding y values with some added noise
    y = np.sin(x) + np.random.normal(0, 0.1, 100)  # sin(x) function with noise

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Bumpy Line", color="blue")

    # Adding title and labels
    plt.title("Bumpy Line Plot")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    # Show legend
    plt.legend()

    # Display the plot
    st.pyplot(plt)












with tab2:
   st.subheader("Feedback Classification")


with tab3:
   st.subheader("Improvement Suggestions")
   
with tab4:
   st.subheader("Rating & Sentiment Analysis")
   
with tab5:
   st.subheader("GPT-4 Feedback Summary")
   
with tab6:
   st.subheader("Word Cloud")
   
with tab7:
   st.subheader("Dataframe")
   
with tab8:
   st.subheader("About")


