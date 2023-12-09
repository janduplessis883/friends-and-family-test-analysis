import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime
from datetime import date


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

st.sidebar.title("Menu")
page = st.sidebar.selectbox(
    "Choose an option",
    [
        "Monthly Rating & Count",
        "Feedback Classification",
        "Improvement Suggestions",
        "Rating & Sentiment Analysis Correlation",
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
if page == "Monthly Rating & Count":
    st.markdown("### Friends & Family Test (FFT) Dashboard")
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
                linewidth=6,
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
                    fontsize=16,  # Adjust this value as needed
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
        st.markdown(f"# {filtered_data.shape[0]}")
        st.write("Total Responses")

    # Plotting the line plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(
        data=daily_count_df, x="Date", y="Daily Count", color="#168aad", linewidth=2
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
        data=monthly_count_filtered_df, x="Month", y="Monthly Count", color="#168aad"
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
    # Create two columns
    col1, col2 = st.columns(2)

# == Rating & Sentiment Analysis Correlation ===============================================
elif page == "Rating & Sentiment Analysis Correlation":
    st.subheader("Rating & Sentiment Analysis Correlation")

    plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
    scatter_plot = sns.scatterplot(
        data=filtered_data,
        x="rating_score",
        y="sentiment_score",
        hue="sentiment",
        s=55,
    )

    # Setting x-axis ticks to 1, 2, 3, 4, 5
    scatter_plot.set_xticks([0.5, 1, 2, 3, 4, 5])
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.6)

    scatter_plot.spines["left"].set_visible(False)
    scatter_plot.spines["top"].set_visible(False)
    scatter_plot.spines["right"].set_visible(False)
    st.pyplot(plt)

    st.write(
        """The plot maps 'rating_score' along the x-axis and 'sentiment_score' along the y-axis. Points on the scatter plot are color-coded to represent three categories of sentiment: positive (blue), neutral (orange), and negative (green). Most of the data points appear to be concentrated at the higher end of the rating scale (closer to 5.0), suggesting a large number of positive sentiment scores. The spread and density of points suggest that higher rating scores correlate with more positive sentiment."""
    )
# == Feedback Classification ==========================================================
elif page == "Feedback Classification":
    st.subheader("Feedback Classification")
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

# == Word Count ==========================================================
elif page == "Word Cloud":
    try:
        st.subheader("Feedback Word Cloud")
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
    st.write("The data below is filtered based on the date range selected above.")

    # Display the filtered DataFrame
    st.write(filtered_data)

# == About ==========================================================
elif page == "About":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Friends & Family Test (FFT) Dashboard")
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
    col1, col2 = st.columns(2)

    # Use 'col1' to display content in the first column
    with col1:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/about.png?raw=true",
            width=250,
        )

    # Use 'col2' to display content in the second column
    with col2:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/hf-logo-with-title.png?raw=true",
            width=200,
        )
        st.markdown(
            "**Text Classification** and **Sentiment Analysis** by Huggingface.co"
        )

# == Improvement Suggestions ==========================================================
elif page == "Improvement Suggestions":
    st.subheader("Improvement Suggestions")

    # Calculate value counts
    label_counts = filtered_data["improvement_labels"].value_counts(
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
            or label == "No Improvment Suggestion"
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
    filtered_classes = filtered_data[
        filtered_data["improvement_labels"].isin(selected_ratings)
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
