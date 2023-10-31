import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


@st.cache_data  # This decorator will help you cache the data
def load_data():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
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
        "Monthly Rating",
        "Monthly Count",
        "Sentiment Analysis Histogram",
        "Rating & Sentiment Correlation",
        "Text Classification",
        "Word Cloud",
        "Show Raw Data",
        "About",
    ],
)


# Display content based on the selected page
if page == "Monthly Rating":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )

    # Plot monthly averages
    st.subheader("Monthly Average Rating")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        x="Month", y="Average Rating", data=monthly_avg_df, color="#a54b49", linewidth=3
    )
    plt.title("Monthly Average Rating")
    plt.xticks(rotation=45)

    # Annotate each point with its value
    for index, row in monthly_avg_df.iterrows():
        ax.annotate(
            f"{row['Average Rating']:.2f}",
            (row["Month"], row["Average Rating"]),
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to points (x,y)
            ha="center",
        )  # horizontal alignment can be left, right or center

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.write("Rating Scale 1 - 5")

elif page == "Monthly Count":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2b.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Monthly Count")
    # Resample and count the entries per day
    monthly_count = data_time.resample("M").size()

    # Reset index to convert Series to DataFrame
    monthly_count = monthly_count.reset_index(name="entry_count")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=monthly_count, x="time", y="entry_count", color="#7c7c6d")

    # Customizing x-axis labels
    n = len(monthly_count["time"])
    tick_frequency = n // 4  # Adjust as needed
    plt.xticks(
        ticks=range(0, n, tick_frequency),
        labels=[
            monthly_count["time"].iloc[i].strftime("%Y-%m-%d")
            for i in range(0, n, tick_frequency)
        ],
        rotation=45,
    )
    plt.title("Friend & Family Test Responses per Month")

    # Show the plot in Streamlit
    st.pyplot(fig)

elif page == "Sentiment Analysis Histogram":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2b.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Sentiment Analysis Histogram")

    if st.checkbox("Review Last Month Only"):
        # Filter the data for 'Negative' and 'Positive' values in the 'score3' column
        data["time"] = pd.to_datetime(data["time"])
        last_30_days = data[data["time"] > data["time"].max() - pd.Timedelta(days=30)]
        data_positive = last_30_days[last_30_days["label3"] == "positive"]["score3"]
        data_negative = last_30_days[last_30_days["label3"] == "negative"]["score3"]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the histograms
        sns.histplot(data_positive, color="#777768", kde=True, label="Positive", ax=ax)
        sns.histplot(data_negative, color="#3e3e33", kde=True, label="Negative", ax=ax)
        # Add title and labels
        plt.title("Sentiment Analysis of FF Test Responses")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        # Show the plot
        st.pyplot(fig)
    else:
        # Filter the data for 'Negative' and 'Positive' values in the 'score3' column
        data_positive = data[data["label3"] == "positive"]["score3"]
        data_negative = data[data["label3"] == "negative"]["score3"]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the histograms
        sns.histplot(
            data_positive, color="#777768", bins=20, kde=True, label="Positive", ax=ax
        )
        sns.histplot(
            data_negative, color="#3e3e33", bins=20, kde=True, label="Negative", ax=ax
        )
        # Add title and labels
        plt.title("Sentiment Analysis of FF Test Responses")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        # Show the plot
        st.pyplot(fig)

elif page == "Rating & Sentiment Correlation":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2b.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Sentiment Anaylsis & Rating Correlation")

    import streamlit as st

    if st.checkbox("Review Last Month Only"):
        data["time"] = pd.to_datetime(data["time"])
        last_30_days = data[data["time"] > data["time"].max() - pd.Timedelta(days=30)]
        correlation = last_30_days["score3"].corr(last_30_days["rating_score"])
        st.write(f"Correlation: {correlation}")
        
        order_legend = ['negative', 'neutral', 'positive']
        custom_palette = {'negative': 'red', 'neutral': 'grey', 'positive': 'green'}
        # Create a Matplotlib figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # First subplot
        sns.scatterplot(x='rating_score', y='score1', data=last_30_days, ax=axes[0], hue_order=order_legend, palette=custom_palette)
        axes[0].set_title('Sentiment Analysis vs Rating Score')

        # Second subplot
        sns.countplot(x='rating_score', hue='label3', data=last_30_days, ax=axes[1])
        axes[1].set_title('Positive/Neutral/Negative Sentiment Analysis Distribution per Rating Score')
        axes[1].legend(title='Sentiment', loc='upper left')

        # Make sure the layout is tight so nothing is cut off
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)
    else:
        correlation = data["score3"].corr(data["rating_score"])
        st.write(f"Correlation: {correlation}")
        order_legend = ['negative', 'neutral', 'positive']
        custom_palette = {'negative': 'red', 'neutral': 'grey', 'positive': 'green'}
        # Create the plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # First subplot
        sns.scatterplot(x='rating_score', y='score1', data=data, ax=axes[0], hue_order=order_legend, palette=custom_palette)
        axes[0].set_title('Sentiment Analysis vs Rating Score')

        # Second subplot
        sns.countplot(x='rating_score', hue='label3', data=data, ax=axes[1])
        axes[1].set_title('Positive/Neutral/Negative Sentiment Analysis Distribution per Rating Score')
        axes[1].legend(title='Sentiment', loc='upper left')

        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)

elif page == "Text Classification":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2b.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Text Classification")

    # Prepare the data
    label_counts = data["label1"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    label_counts2 = data["label2"].value_counts().reset_index()
    label_counts2.columns = ["label2", "count"]

    # Sort the dataframe by count
    label_counts = label_counts.sort_values(by="count", ascending=False)
    label_counts2 = label_counts2.sort_values(by="count", ascending=False)
    # Display DataFrame with index set to 3
    st.write("Text Classification Frequency: Label 1")

    # Plot using seaborn
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=label_counts, y="label", x="count", color="#777768")

    # Display the plot in Streamlit
    st.pyplot(fig3)
    st.write("Text Classification Frequency: Label 2")
    # Plot using seaborn
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=label_counts2, y="label2", x="count", color="#8e8e7f")

    # Display the plot in Streamlit
    st.pyplot(fig4)
    st.subheader("Free Text Response")
    unique_combinations = data[["label1", "label2"]].drop_duplicates()

    # Show all unique combinations
    data["time"] = pd.to_datetime(data["time"])
    last_30_days = data[data["time"] > data["time"].max() - pd.Timedelta(days=30)]
    for index, row in last_30_days.iterrows():
        if type(row['full_text']) != float:
            st.write('✏️ ', row['full_text'])
            st.write(row['label3'], row['score3'], row['label1'], row['score1'], row['label2'], row['score2'])
            st.markdown("---")

elif page == "Word Cloud":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2b.png?raw=true",
        use_column_width=True,
    )
    st.subheader("Word Cloud")
    if st.checkbox("Display Last Month Only"):
        data["time"] = pd.to_datetime(data["time"])
        last_30_days = data[data["time"] > data["time"].max() - pd.Timedelta(days=30)]
        text = " ".join(last_30_days["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="cividis").generate(
            text
        )
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        text = " ".join(data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="cividis").generate(
            text
        )
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)


elif page == "Show Raw Data":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2b.png?raw=true",
        use_column_width=True,
    )
    st.subheader("View Raw Data")
    # Create a slider for selecting the number of rows to display
    # Create a slider for selecting the number of rows to display
    num_rows = st.slider(
        "Select number of rows to display:", min_value=5, max_value=100
    )

    # Create a dropdown for selecting the rating_score to filter by
    selected_rating = st.selectbox("Select rating score to filter by:", [1, 2, 3, 4, 5])

    # Filter the DataFrame based on the selected rating_score
    filtered_data = data[data["rating_score"] == selected_rating]

    # Use the slider's value to display that many rows from the tail of the filtered DataFrame
    if num_rows:
        st.subheader(
            f"Displaying last {num_rows} rows of raw data with rating score {selected_rating}"
        )
        st.write(filtered_data.tail(num_rows))

elif page == "About":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )
    st.subheader("About this Dashboard")
    st.markdown(
        """The Friends & Family Test (FFT) is a widely-used feedback tool that aims to capture patient experiences and gauge the overall quality of healthcare services. By employing text classification and sentiment analysis techniques, we can automatically categorize patient feedback into various themes like service quality, staff behavior, or facility cleanliness. This not only streamlines the process of interpreting large volumes of free-text responses but also provides actionable insights. 
             \nSentiment analysis further enhances this by assigning a polarity score to each response, indicating whether the sentiment is positive, negative, or neutral. This multi-layered approach allows healthcare providers to have a nuanced understanding of patient satisfaction. 
             \nIt identifies areas for improvement, recognizes outstanding service, and ultimately, helps in making data-driven decisions to enhance patient care.
             \n**Sentiment Analysis** - 🤗HuggingFace: `cardiffnlp/twitter-roberta-base-sentiment-latest`
             \n**Text Classification** - 🤗HuggingFace: `SamLowe/roberta-base-go_emotions`
             \n Dashboard by [janduplessis883](https://github.com/janduplessis883)"""
    )
