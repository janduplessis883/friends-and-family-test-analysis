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
import streamlit_shadcn_ui as ui


client = OpenAI()

from utils import *

st.set_page_config(page_title="AI MedReview: FFT")

html = """
<style>
.gradient-text {
    background: linear-gradient(45deg, #284d74, #d8ad45, #ae4f4d);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 2em;
    font-weight: bold;
}
</style>
<div class="gradient-text">AI MedReview: FFT</div>

"""


@st.cache_data(ttl=100)
def load_data():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    df.drop_duplicates(inplace=True)
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

st.sidebar.markdown(html, unsafe_allow_html=True)
st.sidebar.image(
    "images/transparent2.png"
)

@st.cache_data(ttl=100)  # This decorator enables caching for this function
def get_surgery_data(data, selected_surgery):
    # Extracting unique surgery types
    surgery_list = data["surgery"].unique()

    # Filtering the dataset based on the selected surgery type
    surgery_data = data[data["surgery"] == selected_surgery]
    return surgery_data



page = st.sidebar.radio(
    "Select a Page",
    [
        "PCN Dashboard",
        "Surgery Dashboard",
        "Feedback Classification",
        "Improvement Suggestions",
        "Feedback Timeline",
        "Sentiment Analysis",
        "Word Cloud",
        "GPT4 Summary",
        "Dataframe",
        "About",
    ],
)
# Only show the surgery selection if the selected page is not 'Survey Summary' or 'About'
if page not in ["PCN Dashboard", "About"]:
    surgery_list = data["surgery"].unique()
    surgery_list.sort()
    selected_surgery = st.sidebar.selectbox("Select Surgery", surgery_list)

    # Call the function with the selected surgery
    surgery_data = get_surgery_data(data, selected_surgery)
else:
    selected_surgery = 'Earls Court Medical Centre'
    
st.sidebar.container(height=200, border=0)
# Call the function with the selected surgery

surgery_data = get_surgery_data(data, selected_surgery)

st.sidebar.write("")

centered_html = """
    <style>
    .centered {
        text-align: center;
    }
    </style>
    <div class='centered'>
    <img alt="Static Badge" src="https://img.shields.io/badge/github-janduplessis883-%23d0ae57?logo=github&color=%23d0ae57&link=https%3A%2F%2Fgithub.com%2Fjanduplessis883%2Ffriends-and-family-test-analysis">
    </div>
"""


# Using the markdown function with HTML to center the text
st.sidebar.markdown(centered_html, unsafe_allow_html=True)


# Create a date range slider
start_date = surgery_data["time"].dt.date.min()
current_date = date.today()

selected_date_range = st.slider(
    f"**{selected_surgery}**",
    min_value=start_date,
    max_value=current_date,
    value=(start_date, current_date),
    help="Select a start and end date",  # Set default range
)


@st.cache_data(ttl=100)  # This decorator caches the output of this function
def filter_data_by_date_range(data, date_range):
    """
    Filter the provided DataFrame based on a date range.

    Parameters:
    data (DataFrame): The DataFrame to filter.
    date_range (tuple): A tuple of two dates (start_date, end_date).

    Returns:
    DataFrame: Filtered DataFrame.
    """
    # Ensure that the 'time' column is a datetime type
    data["time"] = pd.to_datetime(data["time"], dayfirst=True)

    # Apply the date range filter
    filtered_d = data[
        (data["time"].dt.date >= date_range[0])
        & (data["time"].dt.date <= date_range[1])
    ]
    return filtered_d


# Example usage in your Streamlit app
# surgery_data and selected_date_range should be defined earlier in your app
filtered_data = filter_data_by_date_range(surgery_data, selected_date_range)


# == DASHBOARD ==========================================================================================================
if page == "Surgery Dashboard":
    st.title(f"{selected_surgery}")
    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    # React to the toggle's state

    if toggle:
        st.markdown(
            """1. **Average Monthly Rating (Line Chart)**:
This line chart shows the average rating given by patients each month. The y-axis represents the average rating, and the x-axis represents time. Each point on the line represents the average rating for that month, allowing viewers to track changes over time.
2. **Rating Distribution (Horizontal Bar Chart)**:
The horizontal bar chart below the line chart represents the distribution of ratings across different categories such as 'Extremely likely', 'Likely', 'Neither likely nor unlikely', 'Unlikely', 'Extremely unlikely', and 'Don't know'. The length of each bar correlates with the count of responses in each category.
3. **Daily FFT Responses (Time Series Plot)**:
This time series plot displays the daily count of FFT responses over the same period. The y-axis shows the number of responses, while the x-axis corresponds to the days within each month. Spikes in the graph may indicate specific days when an unusually high number of responses were collected.
4. **Monthly FFT Responses (Bar Chart)**:
The final plot is a vertical bar chart showing the total count of FFT responses collected each month. The y-axis represents the count of responses, and the x-axis indicates the month. Each bar's height represents the total number of responses for that month, providing a clear comparison of month-to-month variation in the volume of feedback."""
        )
    col1, col2 = st.columns([5, 2])

    # Use the columns
    with col1:
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
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                x="time",
                y="rating_score",
                data=monthly_avg_df,
                ax=ax,
                linewidth=4,
                color="#e5c17e",
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
            st.info("No rating available for this date range.")

    with col2:
        ui.metric_card(
            title="Total Responses",
            content=f"{filtered_data.shape[0]}",
            description=f"since {start_date}",
            key="card1",
        )

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
        "Extremely likely": "#112f45",
        "Likely": "#4d9cb9",
        "Neither likely nor unlikely": "#9bc8e3",
        "Unlikely": "#f4ba41",
        "Extremely unlikely": "#ec8b33",
        "Don't know": "#ae4f4d",
    }

    # Set the figure size (width, height) in inches
    st.markdown("---")
    plt.figure(figsize=(12, 4))

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
        bbox_to_anchor=(1.05, 1),
        loc="best",
    )

    # Iterate through the rectangles (bars) of the plot for width annotations
    for p in ax.patches:
        width = p.get_width()
        offset = width * 0.02
        try:
            y = p.get_y() + p.get_height() / 2
            ax.text(
                width + offset,
                y,
                f"{int(width)} / {round((int(width)/filtered_data.shape[0])*100, 1)}%",
                va="center",
                fontsize=10,
            )
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
    st.write("---")

    # Plotting the line plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.lineplot(
        data=daily_count_df, x="Date", y="Daily Count", color="#558387", linewidth=2
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
    st.markdown("---")
    # Resample and count the entries per month from filtered data
    monthly_count_filtered = filtered_data.resample("M", on="time").size()
    monthly_count_filtered_df = monthly_count_filtered.reset_index()
    monthly_count_filtered_df.columns = ["Month", "Monthly Count"]
    monthly_count_filtered_df["Month"] = pd.to_datetime(
        monthly_count_filtered_df["Month"]
    )
    # Create the figure and the bar plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.barplot(
        data=monthly_count_filtered_df, x="Month", y="Monthly Count", color="#aabd3b"
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
    
 

# == Rating & Sentiment Analysis Correlation ======================================================================
elif page == "Sentiment Analysis":
 
    st.title("Sentiment Analysis")
    
    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )

    # React to the toggle's state
    if toggle:
        st.markdown(
            """1. **Scatter Plot (Top Plot)**:
This plot compares patient feedback sentiment scores with feedback rating scores. On the x-axis, we have the rating score, which likely corresponds to a numerical score given by the patient in their feedback, and on the y-axis, we have the sentiment score, which is derived from sentiment analysis of the textual feedback provided by the patient. Each point represents a piece of feedback, categorized as 'positive', 'neutral', or 'negative' sentiment, depicted by different markers. The scatter plot shows a clear positive correlation between the sentiment score and the feedback rating score, especially visible with the concentration of 'positive' sentiment scores at the higher end of the rating score scale, suggesting that more positive text feedback corresponds to higher numerical ratings.
2. **Histogram with a Density Curve (Bottom Left - NEGATIVE Sentiment)**:
This histogram displays the distribution of sentiment scores specifically for negative sentiment feedback. The x-axis represents the sentiment score (presumably on a scale from 0 to 1), and the y-axis represents the count of feedback instances within each score range. The bars show the frequency of feedback at different levels of negative sentiment, and the curve overlaid on the histogram provides a smooth estimate of the distribution. The distribution suggests that most negative feedback has a sentiment score around 0.7 to 0.8.
3. **Histogram with a Density Curve (Bottom Right - POSITIVE Sentiment)**:
Similar to the negative sentiment histogram, this one represents the distribution of sentiment scores for positive sentiment feedback. Here, we see a right-skewed distribution with a significant concentration of feedback in the higher sentiment score range, particularly close to 1.0. This indicates that the positive feedback is often associated with high sentiment scores, which is consistent with the expected outcome of sentiment analysis.
4. **View Patient Feedback (Multi-Select Input)**:
Select Patient feedback to review, this page only displays feedback that on Sentiment Analysis scored **NEGATIVE > Selected Value (using slider)**, indicating negative feedback despite rating given by the patient. It is very important to review feedback with a high NEG sentiment analysis. In this section both feedback and Improvement Suggestions are displayed to review them in context, together with the automated category assigned by our machine learning model."""
        )

    # Data for plotting
    labels = "Positive", "Neutral", "Negative"
    sizes = sentiment_totals(filtered_data)
    colors = ["#6894a8", "#eee8d6", "#ae4f4d"]
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
    st.pyplot(fig)
    st.markdown("---")

    # Resample and count the entries per month from filtered data
    weekly_sent = filtered_data.resample("W", on="time")[
        "neg", "pos", "neu", "compound"
    ].mean()
    weekly_sent_df = weekly_sent.reset_index()
    weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]
    weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])

    @st.cache_data(ttl=100) # This decorator caches the output of this function
    def calculate_weekly_sentiment(data):
        """
        Calculate the weekly sentiment averages from the given DataFrame.

        Parameters:
        data (DataFrame): The DataFrame containing sentiment scores and time data.

        Returns:
        DataFrame: A DataFrame with weekly averages of sentiment scores.
        """
        # Resample the data to a weekly frequency and calculate the mean of sentiment scores
        weekly_sent = data.resample("W", on="time")[
            "neg", "pos", "neu", "compound"
        ].mean()

        # Reset the index to turn the 'time' index into a column and rename columns
        weekly_sent_df = weekly_sent.reset_index()
        weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]

        # Convert the 'Week' column to datetime format
        weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])

        return weekly_sent_df

    weekly_sentiment = calculate_weekly_sentiment(filtered_data)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(
        data=weekly_sentiment,
        x="Week",
        y="neu",
        color="#eee8d6",
        label="Neutral",
        linewidth=2,
    )

    sns.lineplot(
        data=weekly_sentiment,
        x="Week",
        y="pos",
        color="#6894a8",
        label="Positive",
        linewidth=2,
    )
    sns.lineplot(
        data=weekly_sentiment,
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
    st.markdown("---")
    
    @st.cache_data(ttl=100)  # This decorator caches the output of this function
    def process_sentiment_data(data):
        """
        Process the sentiment data to calculate the percentage of positive, negative,
        and neutral sentiments for each date.

        Parameters:
        data (DataFrame): The DataFrame containing sentiment data and time.

        Returns:
        DataFrame: A DataFrame with the processed sentiment data.
        """
        # Ensure 'time' column is in datetime format and create a 'date' column
        data["date"] = pd.to_datetime(data["time"]).dt.date

        pos_list, neg_list, neu_list, date_list = [], [], [], []

        # Calculate the sentiment percentages
        for i in data["date"].unique():
            temp = data[data["date"] == i]
            positive_temp = temp[temp["sentiment"] == "positive"]
            negative_temp = temp[temp["sentiment"] == "negative"]
            neutral_temp = temp[temp["sentiment"] == "neutral"]

            pos_list.append((positive_temp.shape[0] / temp.shape[0]) * 100)
            neg_list.append((negative_temp.shape[0] / temp.shape[0]) * 100)
            neu_list.append((neutral_temp.shape[0] / temp.shape[0]) * 100)
            date_list.append(str(i))

        # Create a new DataFrame with the calculated data
        new_data = pd.DataFrame(
            {"date": date_list, "pos": pos_list, "neg": neg_list, "neu": neu_list}
        )

        # Convert 'date' to datetime and normalize the sentiment columns
        new_data["date"] = pd.to_datetime(new_data["date"])
        new_data["total"] = new_data[["neg", "pos", "neu"]].sum(axis=1)
        new_data["neg"] /= new_data["total"]
        new_data["pos"] /= new_data["total"]
        new_data["neu"] /= new_data["total"]

        # Convert 'date' back to string for plotting and sort the DataFrame
        new_data["date"] = new_data["date"].dt.strftime("%Y-%m-%d")
        new_data = new_data.sort_values("date")

        return new_data

    # Example usage in your Streamlit app
    # Assume filtered_data is defined earlier in your app
    new = process_sentiment_data(filtered_data)

    # Get the date_list for x-axis ticks
    date_list = new["date"]

    # Create the bottom parameters for stacking
    bottom_pos = new["neg"]
    bottom_neu = new["neg"] + new["pos"]

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(16, 5))

    # Plot each sentiment as a layer in the stacked bar
    ax.bar(new["date"], new["neg"], label="Negative", color="#ae4f4d", alpha=1)
    ax.bar(
        new["date"],
        new["pos"],
        bottom=bottom_pos,
        label="Positive",
        color="#6894a8",
        alpha=0.9,
    )
    ax.bar(
        new["date"],
        new["neu"],
        bottom=bottom_neu,
        label="Neutral",
        color="#eee8d6",
        alpha=0.6,
    )

    # Set grid, spines and annotations as before
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # Rotate the x-axis dates for better readability
    plt.xticks(rotation=90, fontsize=8)  # Set x-tick font size

    # Add legend
    plt.legend()
    # Redraw the figure to ensure the formatter is applied
    # Set title to the right
    ax_title = ax.set_title("Daily Sentiment", loc="right")
    ax_title.set_position((1.02, 1))  # Adjust title position
    fig.canvas.draw()

    # Remove xlabel as it's redundant with the dates
    plt.xlabel("Unique Days")
    plt.ylabel("Sentiment Analysis")
    # Apply tight layout and display plot
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

    st.markdown("---")

    palette_colors = {
        "positive": "#4187aa",
        "neutral": "#d8ae46",
        "negative": "#be6933",
    }
    plt.figure(figsize=(12, 4))  # You can adjust the figure size as needed
    scatter_plot = sns.scatterplot(
        data=filtered_data,
        y="rating_score",
        x="compound",
        hue="sentiment",
        s=65,
        palette=palette_colors,
        marker="o",
    )

    # Setting x-axis ticks to 1, 2, 3, 4, 5
    # Define the color palette as a dictionary

    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.6)

    scatter_plot.spines["left"].set_visible(False)
    scatter_plot.spines["top"].set_visible(False)
    scatter_plot.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(plt)

    # Negative sentiment plot
    neg_sentiment = filtered_data[filtered_data["compound"] < 0]
    slider_start_point = neg_sentiment["neg"].min()
    if slider_start_point == 0:
        slider_start = 0.4
    else:
        slider_start = slider_start_point

    # The value parameter is set to slider_end, which is the maximum value
    slider_value = st.slider(
        label="Select Negative Sentiment Analysis threshold:",
        min_value=slider_start,
        max_value=1.0,
        value=0.9,  # Set initial value to the max value
        step=0.1,
    )

    # Create two columns
    col1, col2 = st.columns(2)

    # Content for the first column
    with col1:

        fig, ax = plt.subplots(figsize=(5, 2))
        sns.histplot(data=neg_sentiment, x="neg", color="#be6933", kde=True, bins=10)
        # Set grid, spines and annotations as before
        # Add a vertical red line at sentiment score of 0.90
        plt.axvline(x=slider_value, color="#ae4f4d", linestyle="-", linewidth=4)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.xlabel("Sentiment Score")
        plt.title("Negative Sentiment")
        st.pyplot(plt)  # Display the plot in Streamlit

    # Content for the second column
    with col2:
        # Positive sentiment plot
        pos_sentiment = filtered_data[filtered_data["compound"] > 0]
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.histplot(data=pos_sentiment, x="pos", color="#4187aa", kde=True)
        # Set grid, spines and annotations as before
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.xlabel("Sentiment Score")
        plt.title("Positive Sentiment")
        st.pyplot(plt)  # Display the plot in Streamlit

    st.markdown("---")
    st.subheader("View Patient Feedback")

    # Create the slider

    # View SELECTED Patient Feedback with Sentiment Analaysis NEG >= 0.5
    selected_feedback = filtered_data[
        (filtered_data["neg"] >= 0.3)
        
    ].sort_values(by="neg", ascending=False)

    class_list = list(selected_feedback["feedback_labels"].unique())
    cleaned_class_list = [x for x in class_list if not pd.isna(x)]
    selected_ratings = st.multiselect(
        f"Viewing Feedback with Sentiment Analysis *NEG > {slider_value}:",
        cleaned_class_list,
        default=cleaned_class_list,
    )

    # Filter the data based on the selected classifications
    filtered_classes = selected_feedback[
        selected_feedback["feedback_labels"].isin(selected_ratings)
    ]

    if not selected_ratings:
        ui.badges(
            badge_list=[("Please select at least one classification.", "outline")],
            class_name="flex gap-2",
            key="badges10",
        )

    else:
        for rating in selected_ratings:
            specific_class = filtered_classes[
                filtered_classes["feedback_labels"] == rating
            ]
            st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
            for _, row in specific_class.iterrows():
                text = row["free_text"]
                do_better = row["do_better"]
                sentiment_score = row["sentiment_score"]

                # Check if the text is valid and not neutral or nan
                if str(text) not in ["nan"]:
                    st.markdown("🗣️ " + str(text))
                    if str(do_better) not in ["nan"]:
                        st.markdown("🔧 " + str(do_better))
                    if str(sentiment_score).lower() not in [
                        "nan",
                    ]:
                        st.markdown("`Neg: " + str(sentiment_score) + "`")


# == Feedback Classification ========================================================================================
elif page == "Feedback Classification":
    st.title("Feedback Classification")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """1. **Bar Chart**:
This bar chart illustrates the range of emotions captured in the FFT feedback, as categorized by a sentiment analysis model trained on the `go_emotions` dataset. Each bar represents one of the 27 emotion labels that the model can assign, showing how often each emotion was detected in the patient feedback.
The **'neutral' category**, which has been assigned the most counts, includes instances where patients did not provide any textual feedback, defaulting to a 'neutral' classification. Other emotions, such as 'admiration' and 'approval', show varying lower counts, reflecting the variety of sentiments expressed by patients regarding their care experiences.

2. **Multi-select Input Field**:
Below the chart is a multi-select field where you can choose to filter and review the feedback based on these emotion labels. This feature allows you to delve deeper into the qualitative data, understanding the nuances behind the ratings patients have given and potentially uncovering areas for improvement in patient experience."""
        )

    # Calculate value counts
    label_counts = filtered_data["feedback_labels"].value_counts(
        ascending=False
    )  # Use ascending=True to match the order in your image

    # Convert the Series to a DataFrame
    label_counts_df = label_counts.reset_index()
    label_counts_df.columns = ["Feedback Classification", "Counts"]

    # Define the palette conditionally based on the category names
    palette = [
        "#aec867" if (label == "Overall Patient Satisfaction") else "#62899f"
        for label in label_counts_df["Feedback Classification"]
    ]

    # Create a Seaborn bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x="Counts", y="Feedback Classification", data=label_counts_df, palette=palette
    )
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    # Adding titles and labels for clarity
    plt.title("Counts of Feedback Classification")
    plt.xlabel("Counts")
    plt.ylabel("")

    # Streamlit function to display matplotlib figures
    st.pyplot(plt)
    st.markdown("---")
    # View Patient Feedback
    st.subheader("View Patient Feedback")
    class_list = list(filtered_data["feedback_labels"].unique())
    cleaned_class_list = [x for x in class_list if not pd.isna(x)]
    selected_ratings = st.multiselect("Select Feedback Categories:", cleaned_class_list)

    # Filter the data based on the selected classifications
    filtered_classes = filtered_data[
        filtered_data["feedback_labels"].isin(selected_ratings)
    ]

    if not selected_ratings:
        ui.badges(
            badge_list=[("Please select at least one classification.", "outline")],
            class_name="flex gap-2",
            key="badges10",
        )
    else:
        for rating in selected_ratings:
            specific_class = filtered_classes[
                filtered_classes["feedback_labels"] == rating
            ]
            st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
            for text in specific_class[
                "free_text"
            ]:  # Assuming 'free_text' is the column with the text you want to display
                if str(text).lower() != "nan" and str(text).lower() != "neutral":
                    st.write("- " + str(text))

# == Word Cloud ==========================================================
elif page == "Word Cloud":
    
    try:
        st.header("Word Cloud")
 
        toggle = ui.switch(
            default_checked=False, label="Explain this page.", key="switch_dash"
        )
        if toggle:
            st.markdown(
                """1. The **Feedback Word Cloud**:
    From response to FFT Q1: Please tell us why you feel this way? 
    A **word cloud** is a visual representation of text data where the size of each word indicates its frequency or importance. In a word cloud, commonly occurring words are usually displayed in larger fonts or bolder colors, while less frequent words appear smaller. This makes it easy to perceive the most prominent terms within a large body of text at a glance.
    In the context of patient feedback, a word cloud can be especially useful to quickly identify the key themes or subjects that are most talked about by patients. For example, if many patients mention terms like "waiting times" or "friendly staff," these words will stand out in the word cloud, indicating areas that are notably good or need improvement.  
2. The **Improvement Suggestions Word Cloud** is a creative and intuitive representation of the feedback collected from patients through the Friends and Family Test (FFT). When patients are asked, "Is there anything that would have made your experience better?" their responses provide invaluable insights into how healthcare services can be enhanced."""
            )
        st.subheader("Feedback Word Cloud")
        text = " ".join(filtered_data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    except:
        ui.badges(
            badge_list=[("No Feedback available for this date range.", "outline")],
            class_name="flex gap-2",
            key="badges10",
        )
    try:
        st.subheader("Improvement Suggestions Word Cloud")

        text2 = " ".join(filtered_data["do_better"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Reds").generate(text2)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    except:
        ui.badges(
            badge_list=[
                ("No improvement suggestions available for this date range.", "outline")
            ],
            class_name="flex gap-2",
            key="badges11",
        )

# == Dataframe ==========================================================
elif page == "Dataframe":
    st.title("Dataframe")
  
    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """**Dataframe**:
A dataFrame as a big, organized table full of raw data. It's like a virtual spreadsheet with many rows and columns, where every row represents a single record, and each column stands for a particular variable. If your DataFrame contains all the raw data, it means that it hasn't been processed or filtered - it's the data in its original form as collected.

Each column in a DataFrame has a name, which you can use to locate data more easily. Columns can contain all sorts of data types, including numbers, strings, and dates, and each one typically holds the same kind of data throughout. For instance, one column might hold ages while another lists names, and yet another records dates of visits.

Rows are labeled with an Index, which you can think of as the address of the data. This makes finding specific records simple and fast."""
        )
    st.write("The data below is filtered based on the date range selected above.")

    # Display the filtered DataFrame
    st.dataframe(filtered_data)

# == About ==========================================================
elif page == "About":
    st.title("About")
    # st.image(
    #     "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftestabout.png?raw=true",
    #     use_column_width=True,
    # )

    st.markdown(
        """Welcome to our new dashboard, aimed at enhancing how healthcare providers understand and use patient feedback. This tool focuses on the Friends and Family Test (FFT), which is essential for collecting patients' views on healthcare services. Our approach uses advanced text classification and sentiment analysis to organize and interpret this feedback more effectively.

Here's the core idea: Instead of just counting responses, the dashboard analyzes the sentiments behind them—whether positive, negative, or neutral. It assigns a detailed score to each piece of feedback, allowing for a more nuanced understanding of patient satisfaction. This method helps identify specific areas needing improvement and those that are performing well, based on real patient experiences.

For healthcare providers, this tool offers a more insightful way to look at patient feedback. It doesn’t just provide data; it offers a clearer picture of how patients feel about their care. This can help highlight excellence in services and pinpoint areas for potential improvements.

The data we use comes from a GP surgery in West London, showing how this tool can be applied in a real healthcare setting.

We employ several machine learning techniques for analysis:

1. **Sentiment Analysis:** Using Huggingface's 'cardiffnlp/twitter-roberta-base-sentiment-latest' model, we determine the emotional tone of the feedback.
2. **Text Classification** of Patient Feedback: To categorize feedback into different emotional themes, we use the 'SamLowe/roberta-base-go_emotions' model from Huggingface.
3. **Zero-shot Classification** of Patient Improvement Suggestions: The 'facebook/bart-large-mnli' model helps us identify and classify suggestions for improving patient care, even when the model hasn’t been specifically trained on healthcare data.
4. Visit [**AI MedReview**](https://github.com/janduplessis883/friends-and-family-test-analysis) on GitHub, collaboration welcomed."""
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(y='surgery', data=data, color='#59646b')
    for p in ax.patches:
        width = p.get_width()
        try:
            y = p.get_y() + p.get_height() / 2
            ax.text(
                width + 1,
                y,
                f"{int(width)}",
                va="center",
                fontsize=8,
            )
        except ValueError:
            pass
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    st.pyplot(plt)
    
    
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # Use 'col1' to display content in the first column
    with col1:
        st.image(
            "images/about.png",
            width=200,
        )

    # Use 'col2' to display content in the second column
    with col2:
        st.image(
            "images/hf-logo-with-title.png",
            width=200,
        )
    with col3:
        st.image(
            "images/openai.png",
            width=200,
        )


# == Improvement Suggestions ==========================================================
elif page == "Improvement Suggestions":
    st.title("Improvement Suggestions")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """1. This **horizontal bar chart** provides an analysis of patient feedback addressing areas for potential improvement in healthcare services. Each bar represents a unique category of improvement suggestion derived from patient feedback using zero-shot classification with the `facebook/bart-large-mnli` model. Prior to classification, one-word responses are filtered out to ensure meaningful data is processed.
The category **"No Improvement Suggestion"** includes feedback that did not suggest any specific changes, which could be interpreted as a form of passive satisfaction. Similarly, the **"Overall Patient Satisfaction"** category likely captures comments that are generally positive, without indicating a need for improvement.
The length of each bar signifies the count of feedback entries that fall into the corresponding category, providing clear insight into common themes within patient suggestions. For instance, categories like "Reception Services" and "Ambiance of Facility" appear frequently, indicating areas where patients have more frequently suggested improvements.

2. Below the chart is a **multi-select input field** allowing for a more granular exploration of the feedback. This tool enables users to select specific categories and review the actual comments associated with them, aiding healthcare providers in understanding patient perspectives in greater detail and potentially guiding quality improvement initiatives."""
        )

    improvement_data = filtered_data[
        (filtered_data["improvement_labels"] != "No Improvement Suggestion")
    ]
    # Calculate value counts
    label_counts = improvement_data["improvement_labels"].value_counts(
        ascending=False
    )  # Use ascending=True to match the order in your image

    # Convert the Series to a DataFrame
    label_counts_df = label_counts.reset_index()
    label_counts_df.columns = ["Improvement Labels", "Counts"]

    # Define the palette conditionally based on the category names
    palette = [
        (
            "#d89254"
            if (
                label == "Overall Patient Satisfaction"
                or label == "No Improvement Suggestion"
            )
            else "#ae4f4d"
        )
        for label in label_counts_df["Improvement Labels"]
    ]

    # Create a Seaborn bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x="Counts", y="Improvement Labels", data=label_counts_df, palette=palette
    )

    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    # Adding titles and labels for clarity
    plt.title("Counts of Improvement Catergories")
    plt.xlabel("Counts")
    plt.ylabel("")

    # Streamlit function to display matplotlib figures
    st.pyplot(plt)
    st.markdown("---")
    
    st.subheader("View Patient Improvement Suggestions")
    improvement_list = [label for label in label_counts_df["Improvement Labels"]]

    selected_ratings = st.multiselect("Select Categories:", improvement_list)

    # Filter the data based on the selected classifications
    filtered_classes = improvement_data[
        improvement_data["improvement_labels"].isin(selected_ratings)
    ]

    if not selected_ratings:
        ui.badges(
            badge_list=[("Please select at least one classification.", "outline")],
            class_name="flex gap-2",
            key="badges10",
        )
    else:
        for rating in selected_ratings:
            specific_class = filtered_classes[
                filtered_classes["improvement_labels"] == rating
            ]
            st.subheader(f"{str(rating).capitalize()} ({str(specific_class.shape[0])})")
            for text in specific_class[
                "do_better"
            ]:  # Assuming 'free_text' is the column with the text you want to display
                st.write("- " + str(text))

# == Generate ChatGPT Summaries ==========================================================
elif page == "GPT4 Summary":
    st.title("GPT4 Free-Text Summary")

    toggle = ui.switch(
        default_checked=False, label="Explain this page.", key="switch_dash"
    )
    if toggle:
        st.markdown(
            """**What This Page Offers:**

**Automated Summaries**: Leveraging OpenAI's cutting-edge ChatGPT-4, we transform the Friends & Family Test feedback and improvement suggestions into concise, actionable insights.  
**Time-Specific Insights**: Select the period that matters to you. Whether it's a week, a month, or a custom range, our tool distills feedback relevant to your chosen timeframe.  
**Efficient Meeting Preparations**: Prepare for meetings with ease. Our summaries provide a clear overview of patient feedback, enabling you to log actions and decisions swiftly and accurately.  

**How It Works**:

1. **Select Your Time Period**: Choose the dates that you want to analyze.  
2. **AI-Powered Summarization**: ChatGPT-4 reads through the feedback and suggestions, understanding the nuances and key points.  
3. **Receive Your Summary**: Get a well-structured, comprehensive summary that highlights the core sentiments and suggestions from your patients."""
        )
    st.markdown(
        "**Follow the steps below to Summarise Free-Text with GPT4.**"
    )
    # filtered_data = surgery_data[
    #     (surgery_data["time"].dt.date >= selected_date_range[0])
    #     & (surgery_data["time"].dt.date <= selected_date_range[1])
    # ]
    filtered_data["prompt"] = filtered_data["free_text"].str.cat(
        filtered_data["do_better"], sep=" "
    )
    series = pd.Series(filtered_data["prompt"])
    series.dropna(inplace=True)
    word_series = series.to_list()
    text = " ".join(word_series)


    def call_chatgpt_api(text):
        # Example OpenAI Python library request
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. and expert at summarixing friends and family Test Feedback for a GP Surgery",
                },
                {"role": "user", "content": f"Summarize the following text\n\n{text}"},
            ],
        )

        output = completion.choices[0].message.content
        return output

    # Text input
    user_input = text
    
    # Initial state setup, assuming you have session state handling configured
    if 'pin_verified' not in st.session_state:
        st.session_state.pin_verified = False
    if 'pin_sent' not in st.session_state:
        st.session_state.pin_sent = False

    # Mobile number input field
    mobile_number = st.text_input("Enter your mobile number (including 🇬🇧 +44) to receive a PIN:", "")

    # Function to handle sending the PIN (pseudo-code, replace with your Continguity implementation)
    def send_pin(mobile_number):
        # Your code to send PIN via SMS
        st.session_state.pin_sent = True
        # Display a message or handle the result of sending the PIN
        st.info(f"A PIN has been sent to {mobile_number}.")

    # PIN verification input field
    def verify_pin_input():
        # Function to verify the entered PIN (pseudo-code)
        def verify_pin(pin):
            # Your verification code here
            # If verified:
            st.session_state.pin_verified = True
            st.success("PIN successfully verified!")
            # Else, handle failed verification:
            # st.error("Incorrect PIN.")

        pin = st.text_input("Enter the PIN you received")
        verify_pin_button = st.button("Verify PIN", on_click=verify_pin, args=(pin,))
        if not verify_pin_button:
            st.warning("Please enter the PIN you received to proceed.")

    # Button to send the PIN
    if mobile_number and not st.session_state.pin_sent:
        send_pin_button = st.button("Send PIN", on_click=send_pin, args=(mobile_number,))

    # Display the PIN input field after the PIN has been sent
    if st.session_state.pin_sent and not st.session_state.pin_verified:
        verify_pin_input()

    # Only display the "Generate GPT4 Summary" button if the PIN has been verified
    if st.session_state.pin_verified:
        if st.button("Generate GPT4 Summary"):
            # Your existing code to generate summary
            pass

    # Button to trigger summarization
    #if st.button("Summarize with GPT4"):
        if user_input:
            # Call the function to interact with ChatGPT API
            st.markdown("### Input Text")
            code = text
            st.info(f"{code}")

            # Initiate progress bar
            my_bar = st.progress(0)

            # Simulate a loading process
            for percent_complete in range(100):
                time.sleep(0.2)
                my_bar.progress(percent_complete + 1)

            summary = call_chatgpt_api(user_input)

            # Hide the progress bar after completion
            my_bar.empty()
            st.markdown("---")
            st.markdown("### GPT4 Feedback Summary")
            st.markdown("`Copy GPPT4 Summary as required.`")
            st.write(summary)
            st.download_button(
                "Download GPT-4 Output", summary, help="Download summary as a TXT file."
            )

        else:
            st.write(text)
            ui.badges(
                badge_list=[("Not able to summarise text.", "destructive")],
                class_name="flex gap-2",
                key="badges10",
            )
    else:
        st.image(
            "images/openailogo.png"
        )

# == Full Responses ==========================================================
elif page == "Feedback Timeline":
    st.title("Feedback Timeline")

    
    
    daily_count = filtered_data.resample("D", on="time").size()
    daily_count_df = daily_count.reset_index()
    daily_count_df.columns = ["Date", "Daily Count"]
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(
        data=daily_count_df, x="Date", y="Daily Count", color="#558387", linewidth=2
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
    st.markdown("---")
    st.markdown(f"Showing **{filtered_data.shape[0]}** FFT Responses")
    
    with st.container(height=500, border=True):
        for _, row in filtered_data.iterrows():
            free_text = row["free_text"]
            do_better = row["do_better"]
            time = row["time"]
            rating = row["rating"]

            with st.chat_message("user"):
                st.markdown(f"**{rating}** `{time}`")
                if str(free_text) not in ["nan"]:
                    st.markdown("🗣️ " + str(free_text))
                    if str(do_better) not in ["nan"]:
                        st.markdown("💡 " + str(do_better))
                        
                        
# == Generate ChatGPT Summaries ==========================================================
elif page == "PCN Dashboard":
    
    st.title("Brompton Health PCN")


    #alldata_date_range = filter_data_by_date_range(data, selected_date_range)
    pivot_data = data.pivot_table(index='surgery', columns='rating', aggfunc='size', fill_value=0)
    total_responses_per_surgery = pivot_data.sum(axis=1)

    # Compute the percentage of each rating category for each surgery
    percentage_pivot_data = pivot_data.div(total_responses_per_surgery, axis=0) * 100
    # Define the desired column order based on the rating categories
    column_order = ["Extremely likely", "Likely", "Neither likely nor unlikely", "Unlikely", "Extremely unlikely", "Don't know"]

    # Reorder the columns in the percentage pivot data
    ordered_percentage_pivot_data = percentage_pivot_data[column_order]

    # Create the heatmap with the ordered columns
    plt.figure(figsize=(12, 9))
    ordered_percentage_heatmap = sns.heatmap(ordered_percentage_pivot_data, annot=True, fmt=".1f", cmap="Blues", linewidths=.5)
    plt.title('% Heatmap of Surgery Ratings', fontsize=16)
    plt.ylabel('Surgery', fontsize=12)
    plt.xlabel('Rating (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Display the ordered percentage heatmap
    st.pyplot(plt)
    st.markdown("---")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(y='surgery', data=data, color='#59646b')
    for p in ax.patches:
        width = p.get_width()
        try:
            y = p.get_y() + p.get_height() / 2
            ax.text(
                width + 1,
                y,
                f"{int(width)}",
                va="center",
                fontsize=8,
            )
        except ValueError:
            pass
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.yaxis.grid(False)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.title("Total FFT Responses by Surgery", loc="right")
    plt.tight_layout()
    st.pyplot(plt)
    st.markdown("---")
    
    # Convert 'time' to datetime and extract the date
    data['date'] = pd.to_datetime(data['time']).dt.date

    # Group by the new 'date' column and calculate the mean 'rating_score' for each day
    daily_mean_rating = data.groupby('date')['rating_score'].mean().reset_index()
    # Ensure the 'date' column is in datetime format for resampling
    daily_mean_rating['date'] = pd.to_datetime(daily_mean_rating['date'])

    # Set the 'date' column as the index
    daily_mean_rating.set_index('date', inplace=True)

    # Resample the data by week and calculate the mean 'rating_score' for each week
    weekly_mean_rating = daily_mean_rating['rating_score'].resample('W').mean().reset_index()

    # Create a seaborn line plot for weekly mean rating scores
    fig, ax = plt.subplots(figsize=(12, 7))
    weekly_lineplot = sns.lineplot(x='date', y='rating_score', data=weekly_mean_rating, color="#adbd52", linewidth=2)
    
    plt.xlabel('Week')
    plt.ylabel('Average Rating Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set title to the right
    ax_title = ax.set_title("Mean Weekly Rating Score - Brompton Health PCN", loc="right")

    # Display the line plot
    st.pyplot(plt)
    st.markdown("---")
    
        # Resample and count the entries per month from filtered data
    weekly_sent = data.resample("D", on="time")[
        "neg", "pos", "neu", "compound"
    ].mean()
    weekly_sent_df = weekly_sent.reset_index()
    weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]
    weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])

    @st.cache_data(ttl=100) # This decorator caches the output of this function
    def calculate_weekly_sentiment(data):
        """
        Calculate the weekly sentiment averages from the given DataFrame.

        Parameters:
        data (DataFrame): The DataFrame containing sentiment scores and time data.

        Returns:
        DataFrame: A DataFrame with weekly averages of sentiment scores.
        """
        # Resample the data to a weekly frequency and calculate the mean of sentiment scores
        weekly_sent = data.resample("W", on="time")[
            "neg", "pos", "neu", "compound"
        ].mean()

        # Reset the index to turn the 'time' index into a column and rename columns
        weekly_sent_df = weekly_sent.reset_index()
        weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]

        # Convert the 'Week' column to datetime format
        weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])

        return weekly_sent_df

    weekly_sentiment = calculate_weekly_sentiment(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=weekly_sentiment,
        x="Week",
        y="neu",
        color="#eee8d6",
        label="Neutral",
        linewidth=2,
    )

    sns.lineplot(
        data=weekly_sentiment,
        x="Week",
        y="pos",
        color="#6894a8",
        label="Positive",
        linewidth=2,
    )
    sns.lineplot(
        data=weekly_sentiment,
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
    ax_title = ax.set_title("Mean Weekly Sentiment Analysis - Brompton Health PCN", loc="right")
    ax_title.set_position((1.02, 1))  # Adjust title position

    # Redraw the figure to ensure the formatter is applied
    fig.canvas.draw()

    # Remove xlabel as it's redundant with the dates
    plt.xlabel("Weeks")
    plt.ylabel("Mean Sentiment")
    # Apply tight layout and display plot
    plt.tight_layout()
    st.pyplot(fig)
    
