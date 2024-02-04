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
    font-size: 2em;
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

col1, col2 = st.columns([2, 1])
with col2:
    surgery_list = data["surgery"].unique()
    surgery = st.selectbox("", surgery_list)
    surgery_data = data[(data["surgery"] == surgery)]

    start_date = surgery_data["time"].dt.date.min()
    current_date = date.today()
with col1:
    # Create a date range slider
    selected_date_range = st.slider(
        "",
        min_value=start_date,
        max_value=current_date,
        value=(start_date, current_date),  # Set default range
    )

# Filter the DataFrame based on the selected date range
filtered_data = surgery_data[
    (surgery_data["time"].dt.date >= selected_date_range[0])
    & (surgery_data["time"].dt.date <= selected_date_range[1])
]

# Calculate monthly averages
data_time = load_timedata()

monthly_avg = data_time["rating_score"].resample("M").mean()
monthly_avg_df = monthly_avg.reset_index()
monthly_avg_df.columns = ["Month", "Average Rating"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Dashboard", "Feedback Classification", "Improvement Suggestions", "Rating & Sentiment Analysis", "GPT-4 Feedback Summary", "Word Cloud", "Dataframe", "About"])

with tab1:

    col1, col2, col3 = st.columns([1.5,3,1])

with col1:
    st.markdown("**Dashboard**")
       # Data for plotting
    labels = "Positive", "Neutral", "Negative"
    sizes = sentiment_totals(filtered_data)
    colors = ["#344e65", "#f0e8d2", "#ae4f4d"]
    explode = (0, 0, 0)  # 'explode' the 1st slice (Positive)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    fig.gca().add_artist(centre_circle)

    plt.title("Patient Feedback Sentiment Distribution")
    st.pyplot(fig)
   
with col2:
    
    monthly_avg = data_time["rating_score"].resample("M").mean()
    monthly_avg_df = monthly_avg.reset_index()
    monthly_avg_df.columns = ["Month", "Average Rating"]

    # Add more content to col2 as needed
    daily_count = filtered_data.resample("D", on="time").size()
    daily_count_df = daily_count.reset_index()
    daily_count_df.columns = ["Date", "Daily Count"]
    try:
        # Resample to get monthly average rating
        monthly_avg = filtered_data.resample("M", on="time")["rating_score"].mean()

        # Reset index to make 'time' a column again
        monthly_avg_df = monthly_avg.reset_index()

        # Create a line plot
        fig, ax = plt.subplots(figsize=(16, 3))
        sns.lineplot(
            x="time",
            y="rating_score",
            data=monthly_avg_df,
            ax=ax,
            linewidth=3,
            color="#e85d04",
        )

        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        # Customize the plot - remove the top, right, and left spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Rotate x-axis labels
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Annotate the line graph
        for index, row in monthly_avg_df.iterrows():
            ax.annotate(
                f'{row["rating_score"]:.2f}',
                (row["time"], row["rating_score"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=12,  # Adjust this value as needed
            )

        # Add labels and title
        plt.xlabel("")
        plt.ylabel("Average Rating")
        plt.tight_layout()
        ax_title = ax.set_title(
            "Average Monthly Rating", loc="right"
        )  # loc parameter aligns the title
        ax_title.set_position(
            (1, 1)
        )  # Adjust these values to align your title as needed
        # Display the plot in Streamlit
        st.pyplot(fig)
    except:
        st.warning("No rating available for this date range.")
        
    st.write("")
    
    order = [
        "Extremely likely",
        "Likely",
        "Neither likely nor unlikely",
        "Unlikely",
        "Extremely unlikely",
        "Don't know",
    ]

    palette = {
        "Extremely likely": "#003f5c",
        "Likely": "#58508d",
        "Neither likely nor unlikely": "#bc5090",
        "Unlikely": "#dd5182",
        "Extremely unlikely": "#ff6e54",
        "Don't know": "#ffa600",
    }

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(16, 3))

    # Create the countplot
    sns.countplot(data=filtered_data, y="rating", order=order, palette=palette)
    ax = plt.gca()

    # Remove y-axis labels
    ax.set_yticklabels([])

    # Create a custom legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(color=color, label=label) for label, color in palette.items()
    ]
    plt.legend(
        handles=legend_patches,
        title="Rating Categories",
        bbox_to_anchor=(1, 1),
        loc="best",
    )
    
    # Iterate through the rectangles (bars) of the plot for width annotations
    for p in ax.patches:
        width = p.get_width()
        try:
            y = p.get_y() + p.get_height() / 2
            ax.text(width + 1, y, f"{int(width)}", va="center", fontsize=10)
        except ValueError:
            pass

    # Adjust plot appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    plt.xlabel("Count")
    plt.ylabel("Rating")
    plt.tight_layout()
    st.pyplot(plt)
    st.write("")

    # Create Sentiment Analaysis Plot
    # Resample and count the entries per month from filtered data
    weekly_sent = filtered_data.resample("W", on="time")[
        "neg", "pos", "neu", "compound"
    ].mean()
    weekly_sent_df = weekly_sent.reset_index()
    weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]
    weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.lineplot(
        data=weekly_sent_df,
        x="Week",
        y="neu",
        color="#f0e8d2",
        label="Neutral",
        linewidth=2,
    )

    sns.lineplot(
        data=weekly_sent_df,
        x="Week",
        y="pos",
        color="#4c91b0",
        label="Positive",
        linewidth=2,
    )
    sns.lineplot(
        data=weekly_sent_df,
        x="Week",
        y="neg",
        color="#ae4f4d",
        label="Negative",
        linewidth=2,
    )

    # Set grid, spines and annotations as before
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set title to the right
    ax_title = ax.set_title("Mean Weekly Sentiment Analysis", loc="right")
    ax_title.set_position((1.02, 1))  # Adjust title position

    # Redraw the figure to ensure the formatter is applied
    fig.canvas.draw()

    # Remove xlabel as it's redundant with the dates
    plt.xlabel("Weeks")
    plt.ylabel("Mean Sentiment")
    # Apply tight layout and display plot
    plt.tight_layout()
    st.pyplot(fig)

    st.write("")
    # Plotting the line plot
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.lineplot(
        data=daily_count_df, x="Date", y="Daily Count", color="#489fb5", linewidth=2
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(False)

    # Customizing the x-axis labels for better readability
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax_title = ax.set_title(
        "Daily FFT Responses", loc="right"
    )  # loc parameter aligns the title
    ax_title.set_position((1, 1))  # Adjust these values to align your title as needed
    plt.xlabel("")
    plt.tight_layout()
    st.pyplot(fig)
    st.write("")
    # Resample and count the entries per month from filtered data
    monthly_count_filtered = filtered_data.resample("M", on="time").size()
    monthly_count_filtered_df = monthly_count_filtered.reset_index()
    monthly_count_filtered_df.columns = ["Month", "Monthly Count"]
    monthly_count_filtered_df["Month"] = pd.to_datetime(
        monthly_count_filtered_df["Month"]
    )
    # Create the figure and the bar plot
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.barplot(
        data=monthly_count_filtered_df, x="Month", y="Monthly Count", color="#489fb5"
    )

    # Set grid, spines and annotations as before
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Annotate bars with the height (monthly count)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # Set title to the right
    ax_title = ax.set_title("Monthly FFT Responses", loc="right")
    ax_title.set_position((1.02, 1))  # Adjust title position

    # Redraw the figure to ensure the formatter is applied
    fig.canvas.draw()

    # Remove xlabel as it's redundant with the dates
    plt.xlabel("")

    # Apply tight layout and display plot
    plt.tight_layout()
    st.pyplot(fig)

with col3:
    st.write('Here')









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


