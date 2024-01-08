import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime
from datetime import date
from matplotlib.patches import Patch
import time  # Importing time for simulating a time-consuming process
from openai import OpenAI
from streamlit_extras.buy_me_a_coffee import button

client = OpenAI()

from friendsfamilytest.utils import *


# Load the dataframe
def load_data():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df


data = load_data()


def load_timedata():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df


# Calculate monthly averages
data_time = load_timedata()

monthly_avg = data_time["rating_score"].resample("M").mean()
monthly_avg_df = monthly_avg.reset_index()
monthly_avg_df.columns = ["Month", "Average Rating"]

html = """
<style>
.gradient-text {
    background: linear-gradient(45deg, #e16d33, #579cc4, #284d74, #4775a7);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 3em;
    font-weight: bold;
}
</style>
<div class="gradient-text">Healthcare Quality Insights: FFT</div>
"""
# Render the HTML in the Streamlit app
st.markdown(html, unsafe_allow_html=True)
st.sidebar.title("Menu")
page = st.sidebar.selectbox(
    "Choose an option",
    [
        "Dashboard",
        "Feedback Classification",
        "Improvement Suggestions",
        "Rating & Sentiment Analysis Correlation",
        "GPT-4 Feedback Summary",
        "Word Cloud",
        "View Dataframe",
        "About",
    ],
)

# Define start date and current date
start_date = date(2023, 7, 13)
current_date = date.today()


# Create a date range slider
selected_date_range = st.slider(
    "Select a date range",
    min_value=start_date,
    max_value=current_date,
    value=(start_date, current_date),  # Set default range
)

# Filter the DataFrame based on the selected date range
filtered_data = data[
    (data["time"].dt.date >= selected_date_range[0])
    & (data["time"].dt.date <= selected_date_range[1])
]


# == DASHBOARD ================================================================
if page == "Dashboard":
    st.subheader("Friends & Family Test (FFT) Dashboard")
    toggle = st.checkbox("Explain this page?")
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
    col1, col2 = st.columns([5, 1])

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
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.lineplot(
                x="time",
                y="rating_score",
                data=monthly_avg_df,
                ax=ax,
                linewidth=4,
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

    with col2:
        st.text("")
        st.text("")
        st.metric("Total Responses", filtered_data.shape[0])

    order = [
        "Extremely likely",
        "Likely",
        "Neither likely nor unlikely",
        "Unlikely",
        "Extremely unlikely",
        "Don't know",
    ]

    palette = {
        "Extremely likely": "#6a994e",
        "Likely": "#A7C957",
        "Neither likely nor unlikely": "#219ebc",
        "Unlikely": "#ffb700",
        "Extremely unlikely": "#bc4749",
        "Don't know": "#F2E8CF",
    }

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(12, 3))

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
        y = p.get_y() + p.get_height() / 2
        ax.text(width + 1, y, f"{int(width)}", va="center", fontsize=10)

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

    st.markdown("**Sentiment Analysis**")
    # Create Sentiment Analaysis Plot
    # Resample and count the entries per month from filtered data
    weekly_sent = filtered_data.resample("W", on="time")[
        "neg", "pos", "neu", "compound"
    ].mean()
    weekly_sent_df = weekly_sent.reset_index()
    weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]
    weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])
    fig, ax = plt.subplots(figsize=(12, 4))
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

    filtered_data["date"] = pd.to_datetime(filtered_data["time"]).dt.date
    pos_list = []
    neg_list = []
    neu_list = []
    date_list = []
    for i in filtered_data["date"].unique():
        temp = filtered_data[filtered_data["date"] == i]
        positive_temp = temp[temp["sentiment"] == "positive"]
        negative_temp = temp[temp["sentiment"] == "negative"]
        neutral_temp = temp[temp["sentiment"] == "neutral"]
        pos_list.append((positive_temp.shape[0] / temp.shape[0]) * 100)
        neg_list.append((negative_temp.shape[0] / temp.shape[0]) * 100)
        neu_list.append((neutral_temp.shape[0] / temp.shape[0]) * 100)
        date_list.append(str(i))

    dict = {"date": date_list, "pos": pos_list, "neg": neg_list, "neu": neu_list}
    new = pd.DataFrame(dict)

    new["date"] = pd.to_datetime(new["date"])

    # Normalize the sentiment columns so that they sum up to 1 (or 100%)
    new["total"] = new[["neg", "pos", "neu"]].sum(axis=1)
    new["neg"] /= new["total"]
    new["pos"] /= new["total"]
    new["neu"] /= new["total"]

    # Convert 'date' to string for plotting purposes
    new["date"] = new["date"].dt.strftime("%Y-%m-%d")

    # Sort by date if not already
    new = new.sort_values("date")

    # Get the date_list for x-axis ticks
    date_list = new["date"]

    # Create the bottom parameters for stacking
    bottom_pos = new["neg"]
    bottom_neu = new["neg"] + new["pos"]

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot each sentiment as a layer in the stacked bar
    ax.bar(new["date"], new["neg"], label="Negative", color="#ae4f4d", alpha=1)
    ax.bar(
        new["date"],
        new["pos"],
        bottom=bottom_pos,
        label="Positive",
        color="#437e97",
        alpha=0.9,
    )
    ax.bar(
        new["date"],
        new["neu"],
        bottom=bottom_neu,
        label="Neutral",
        color="#f0e8d2",
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

    st.markdown("**Response Count**")
    # Plotting the line plot
    fig, ax = plt.subplots(figsize=(12, 3))
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

    # Resample and count the entries per month from filtered data
    monthly_count_filtered = filtered_data.resample("M", on="time").size()
    monthly_count_filtered_df = monthly_count_filtered.reset_index()
    monthly_count_filtered_df.columns = ["Month", "Monthly Count"]
    monthly_count_filtered_df["Month"] = pd.to_datetime(
        monthly_count_filtered_df["Month"]
    )
    # Create the figure and the bar plot
    fig, ax = plt.subplots(figsize=(12, 3))
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


# == Rating & Sentiment Analysis Correlation ===============================================
elif page == "Rating & Sentiment Analysis Correlation":
    st.subheader("Rating & Sentiment Analysis Correlation")
    toggle = st.checkbox("Explain this page?")

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
    colors = ["#5385a6", "#f0e8d2", "#ae4f4d"]
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

    # Resample and count the entries per month from filtered data
    weekly_sent = filtered_data.resample("W", on="time")[
        "neg", "pos", "neu", "compound"
    ].mean()
    weekly_sent_df = weekly_sent.reset_index()
    weekly_sent_df.columns = ["Week", "neg", "pos", "neu", "compound"]
    weekly_sent_df["Week"] = pd.to_datetime(weekly_sent_df["Week"])
    fig, ax = plt.subplots(figsize=(12, 4))
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

    palette_colors = {
        "positive": "#4187aa",
        "neutral": "#d8ae46",
        "negative": "#be6933",
    }
    plt.figure(figsize=(12, 4))  # You can adjust the figure size as needed
    scatter_plot = sns.scatterplot(
        data=filtered_data,
        y="rating_score",
        x="sentiment_score",
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

    # Create two columns
    col1, col2 = st.columns(2)

    # Content for the first column
    with col1:
        # Negative sentiment plot
        neg_sentiment = filtered_data[filtered_data["sentiment"] == "negative"]
        slider_start_point = neg_sentiment["sentiment_score"].min() - 0.02
        if slider_start_point == 0:
            slider_start = 0.5
        else:
            slider_start = slider_start_point

        fig, ax = plt.subplots(figsize=(5, 2))
        sns.histplot(data=neg_sentiment, x="sentiment_score", color="#be6933", kde=True)
        # Set grid, spines and annotations as before
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
        pos_sentiment = filtered_data[filtered_data["sentiment"] == "positive"]
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.histplot(data=pos_sentiment, x="sentiment_score", color="#4187aa", kde=True)
        # Set grid, spines and annotations as before
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.xlabel("Sentiment Score")
        plt.title("Positive Sentiment")
        st.pyplot(plt)  # Display the plot in Streamlit

    st.subheader("View Patient Feedback")

    # Create the slider
    try:
        # The value parameter is set to slider_end, which is the maximum value
        slider_value = st.slider(
            label="Select Negative Sentiment Analysis threshold:",
            min_value=slider_start,
            max_value=1.0,
            value=0.9,  # Set initial value to the max value
            step=0.02,
        )
    except Exception as e:
        # This will catch any exceptions and display an info message
        st.info("No value to select. Please check the slider configuration.")
        # Optionally, you can also display the exception message
        st.error(f"An error occurred: {e}")

    # View SELECTED Patient Feedback with Sentiment Analaysis NEG >= 0.5
    selected_feedback = filtered_data[
        (filtered_data["sentiment"] == "negative")
        & (filtered_data["sentiment_score"] >= slider_value)
    ].sort_values(by="sentiment_score", ascending=False)

    class_list = list(selected_feedback["classif"].unique())
    selected_ratings = st.multiselect(
        f"Viewing Feedback with Sentiment Analysis *NEG > {slider_value}:",
        class_list,
        default=class_list,
    )

    # Filter the data based on the selected classifications
    filtered_classes = selected_feedback[
        selected_feedback["classif"].isin(selected_ratings)
    ]

    if not selected_ratings:
        st.warning("Please select at least one classification.")
    else:
        for rating in selected_ratings:
            specific_class = filtered_classes[filtered_classes["classif"] == rating]
            st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
            for _, row in specific_class.iterrows():
                text = row["free_text"]
                do_better = row["do_better"]
                improvement_labels = row["improvement_labels"]

                # Check if the text is valid and not neutral or nan
                if str(text).lower() not in ["nan", "neutral", "admiration"]:
                    st.markdown("🗣️ " + str(text))
                    if str(do_better).lower() not in ["nan", "neutrall", "admiration"]:
                        st.markdown("🔧 " + str(do_better))
                    if str(improvement_labels).lower() not in [
                        "nan",
                        "neutral",
                        "admiration",
                    ]:
                        st.markdown("- `" + str(improvement_labels) + "`")


# == Feedback Classification ==========================================================
elif page == "Feedback Classification":
    st.subheader("Feedback Classification")

    toggle = st.checkbox("Explain this page?")
    if toggle:
        st.markdown(
            """1. **Bar Chart**:
This bar chart illustrates the range of emotions captured in the FFT feedback, as categorized by a sentiment analysis model trained on the `go_emotions` dataset. Each bar represents one of the 27 emotion labels that the model can assign, showing how often each emotion was detected in the patient feedback.
The **'neutral' category**, which has been assigned the most counts, includes instances where patients did not provide any textual feedback, defaulting to a 'neutral' classification. Other emotions, such as 'admiration' and 'approval', show varying lower counts, reflecting the variety of sentiments expressed by patients regarding their care experiences.

2. **Multi-select Input Field**:
Below the chart is a multi-select field where you can choose to filter and review the feedback based on these emotion labels. This feature allows you to delve deeper into the qualitative data, understanding the nuances behind the ratings patients have given and potentially uncovering areas for improvement in patient experience."""
        )

    # Calculate value counts
    label_counts = filtered_data["classif"].value_counts(
        ascending=False
    )  # Use ascending=True to match the order in your image

    # Convert the Series to a DataFrame
    label_counts_df = label_counts.reset_index()
    label_counts_df.columns = ["Feedback Classification", "Counts"]

    # Define the palette conditionally based on the category names
    palette = [
        "#90e0ef" if (label == "neutral") else "#0096c7"
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

    # View Patient Feedback
    st.subheader("View Patient Feedback")
    class_list = list(filtered_data["classif"].unique())
    selected_ratings = st.multiselect("Select Feedback Categories:", class_list)

    # Filter the data based on the selected classifications
    filtered_classes = filtered_data[filtered_data["classif"].isin(selected_ratings)]

    if not selected_ratings:
        st.warning("Please select at least one classification.")
    else:
        for rating in selected_ratings:
            specific_class = filtered_classes[filtered_classes["classif"] == rating]
            st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
            for text in specific_class[
                "free_text"
            ]:  # Assuming 'free_text' is the column with the text you want to display
                if str(text).lower() != "nan" and str(text).lower() != "neutral":
                    st.write("- " + str(text))

# == Word Cloud ==========================================================
elif page == "Word Cloud":
    try:
        st.subheader("Feedback Word Cloud")
        toggle = st.checkbox("Explain this page?")
        if toggle:
            st.markdown(
                """1. The **Feedback Word Cloud**:
    From response to FFT Q1: Please tell us why you feel this way? 
    A **word cloud** is a visual representation of text data where the size of each word indicates its frequency or importance. In a word cloud, commonly occurring words are usually displayed in larger fonts or bolder colors, while less frequent words appear smaller. This makes it easy to perceive the most prominent terms within a large body of text at a glance.
    In the context of patient feedback, a word cloud can be especially useful to quickly identify the key themes or subjects that are most talked about by patients. For example, if many patients mention terms like "waiting times" or "friendly staff," these words will stand out in the word cloud, indicating areas that are notably good or need improvement..

    2. The **Improvement Suggestions Word Cloud** is a creative and intuitive representation of the feedback collected from patients through the Friends and Family Test (FFT). When patients are asked, "Is there anything that would have made your experience better?" their responses provide invaluable insights into how healthcare services can be enhanced."""
            )
        text = " ".join(filtered_data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    except:
        st.warning("No feedback available for this date range.")
    try:
        st.subheader("Improvement Suggestions Word Cloud")

        text2 = " ".join(filtered_data["do_better"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Reds").generate(text2)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    except:
        st.warning("No improvement suggestions available for this date range.")

# == Dataframe ==========================================================
elif page == "View Dataframe":
    st.subheader("Dataframe")
    toggle = st.checkbox("Explain this page?")
    if toggle:
        st.markdown(
            """**Dataframe**:
A dataFrame as a big, organized table full of raw data. It's like a virtual spreadsheet with many rows and columns, where every row represents a single record, and each column stands for a particular variable. If your DataFrame contains all the raw data, it means that it hasn't been processed or filtered - it's the data in its original form as collected.

Each column in a DataFrame has a name, which you can use to locate data more easily. Columns can contain all sorts of data types, including numbers, strings, and dates, and each one typically holds the same kind of data throughout. For instance, one column might hold ages while another lists names, and yet another records dates of visits.

Rows are labeled with an Index, which you can think of as the address of the data. This makes finding specific records simple and fast."""
        )
    st.write("The data below is filtered based on the date range selected above.")

    # Display the filtered DataFrame
    st.write(filtered_data)

# == About ==========================================================
elif page == "About":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Friends & Family Test (FFT) Dashboard v 2.5")
    st.markdown(
        """Welcome to our new dashboard, aimed at enhancing how healthcare providers understand and use patient feedback. This tool focuses on the Friends and Family Test (FFT), which is essential for collecting patients' views on healthcare services. Our approach uses advanced text classification and sentiment analysis to organize and interpret this feedback more effectively.

Here's the core idea: Instead of just counting responses, the dashboard analyzes the sentiments behind them—whether positive, negative, or neutral. It assigns a detailed score to each piece of feedback, allowing for a more nuanced understanding of patient satisfaction. This method helps identify specific areas needing improvement and those that are performing well, based on real patient experiences.

For healthcare providers, this tool offers a more insightful way to look at patient feedback. It doesn’t just provide data; it offers a clearer picture of how patients feel about their care. This can help highlight excellence in services and pinpoint areas for potential improvements.

The data we use comes from a GP surgery in West London, showing how this tool can be applied in a real healthcare setting.

We employ several machine learning techniques for analysis:

1. **Sentiment Analysis:** Using Huggingface's 'cardiffnlp/twitter-roberta-base-sentiment-latest' model, we determine the emotional tone of the feedback.
2. **Text Classification** of Patient Feedback: To categorize feedback into different emotional themes, we use the 'SamLowe/roberta-base-go_emotions' model from Huggingface.
3. **Zero-shot Classification** of Patient Improvement Suggestions: The 'facebook/bart-large-mnli' model helps us identify and classify suggestions for improving patient care, even when the model hasn’t been specifically trained on healthcare data.

Developed by [janduplessis883](https://github.com/janduplessis883/friends-and-family-test-analysis)
"""
    )

    st.markdown("---")
    st.markdown(
        "![Static Badge](https://img.shields.io/badge/GitHub-janduplessis883-%23aabd3b)  ![Static Badge](https://img.shields.io/badge/Python-3.10.6-%23ae4f4d) ![Static Badge](https://img.shields.io/badge/Telegram-%40jdp145-%2354a7e5?logo=telegram)"
    )

    def buy_coffee():
        button(username="janduplessis883", floating=False, width=221)

    col1, col2 = st.columns(2)

    # Use 'col1' to display content in the first column
    with col1:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/about.png?raw=true",
            width=250,
        )
        buy_coffee()
    # Use 'col2' to display content in the second column
    with col2:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/hf-logo-with-title.png?raw=true",
            width=200,
        )
        st.markdown(
            "**Text Classification**,  **Sentiment Analysis** and **Zero-Shot Classification** by Huggingface.co"
        )
        st.image(
            "https://raw.githubusercontent.com/janduplessis883/friends-and-family-test-analysis/2f41ce13bfe98f24c8a75a875280c4c051e1dd5e/images/powered-by-openai-badge-filled-on-light.svg",
            width=200,
        )
        st.markdown("**GPT-4 Feedback Summary** by OpenAI")


# == Improvement Suggestions ==========================================================
elif page == "Improvement Suggestions":
    st.subheader("Improvement Suggestions")
    toggle = st.checkbox("Explain this page?")
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
        "#ffba08"
        if (
            label == "Overall Patient Satisfaction"
            or label == "No Improvement Suggestion"
        )
        else "#e85d04"
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
    st.subheader("View Patient Improvement Suggestions")
    improvement_list = [label for label in label_counts_df["Improvement Labels"]]

    selected_ratings = st.multiselect("Select Categories:", improvement_list)

    # Filter the data based on the selected classifications
    filtered_classes = improvement_data[
        improvement_data["improvement_labels"].isin(selected_ratings)
    ]

    if not selected_ratings:
        st.warning("Please select at least one classification.")
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
elif page == "GPT-4 Feedback Summary":
    st.subheader("Generate ChatGPT Summaries")
    toggle = st.checkbox("Explain this page?")
    if toggle:
        st.markdown("""soon...""")
    filtered_data = data[
        (data["time"].dt.date >= selected_date_range[0])
        & (data["time"].dt.date <= selected_date_range[1])
    ]
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
                {"role": "user", "content": f"Summarize the follwing text\n\n{text}"},
            ],
        )

        output = completion.choices[0].message.content
        return output

    # Text input
    user_input = text

    # Button to trigger summarization
    if st.button("Summarize with GPT-4"):
        if user_input:
            # Call the function to interact with ChatGPT API
            st.markdown("### Input Text")
            code = text
            st.info(f"{code}")

            # Initiate progress bar
            my_bar = st.progress(0)

            # Simulate a loading process
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)

            summary = call_chatgpt_api(user_input)

            # Hide the progress bar after completion
            my_bar.empty()
            st.markdown("---")
            st.markdown("### GPT-4 Feedback Summary")
            st.markdown("`Copy GPPT-4 Report as required.`")
            st.write(summary)

        else:
            st.write(text)
            st.warning("Model Error: Not able to summarize feedback.")
