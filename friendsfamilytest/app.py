import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime


def load_data():
    df = pd.read_csv('friendsfamilytest/data/data.csv')
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
        "Sentiment Analysis Histogram",
        "Rating & Sentiment Correlation",
        "Text Classification",
        "Word Cloud",
        "Show Raw Data",
        "About",
    ],
)


# Display content based on the selected page
if page == "Monthly Rating & Count":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )

    # Plot monthly averages
    st.subheader("Average Monthly Rating")
    st.write('''The Friends and Family Test (FFT) is a feedback tool used in the healthcare sector, particularly in the UK's National Health Service (NHS), to help measure patient satisfaction with services. It allows patients to provide feedback on their experience with a particular service, including General Practitioner (GP) surgeries. The test is straightforward, usually asking whether the patient would recommend the service to friends and family if needed. Patients can typically respond with options like "extremely likely," "likely," "neither likely nor unlikely," "unlikely," "extremely unlikely," and "don't know."''')

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        x="Month", y="Average Rating", data=monthly_avg_df, color="#2b6688", linewidth=3
    )
    plt.title("Average Monthly Rating")

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
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#888888')
    st.pyplot(fig)
    
    
    st.subheader("No of Reviews per Month")
    st.write("""
A "FFT (Friends and Family Test) Count per Month" plot is a visual representation used to display the number of responses received for the FFT in a GP surgery over a series of months. This type of plot is particularly useful for understanding patient engagement and the volume of feedback over time. """)
    # Resample and count the entries per day
    monthly_count = data_time.resample("M").size()
    # Reset index to convert Series to DataFrame
    monthly_count = monthly_count.reset_index(name="entry_count")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=monthly_count, x="time", y="entry_count", color="#2d668f")

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
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#888888')
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

        order_legend = ["negative", "neutral", "positive"]
        custom_palette = {"negative": "red", "neutral": "grey", "positive": "green"}
        # Create a Matplotlib figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # First subplot
        sns.scatterplot(
            x="rating_score",
            y="score1",
            data=last_30_days,
            ax=axes[0],
            hue_order=order_legend,
            palette=custom_palette,
        )
        axes[0].set_title("Sentiment Analysis vs Rating Score")

        # Second subplot
        sns.countplot(x="rating_score", hue="label3", data=last_30_days, ax=axes[1])
        axes[1].set_title(
            "Positive/Neutral/Negative Sentiment Analysis Distribution per Rating Score"
        )
        axes[1].legend(title="Sentiment", loc="upper left")

        # Make sure the layout is tight so nothing is cut off
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)
    else:
        correlation = data["score3"].corr(data["rating_score"])
        st.write(f"Correlation: {correlation}")
        order_legend = ["negative", "neutral", "positive"]
        custom_palette = {"negative": "red", "neutral": "grey", "positive": "green"}
        # Create the plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # First subplot
        sns.scatterplot(
            x="rating_score",
            y="score1",
            data=data,
            ax=axes[0],
            hue_order=order_legend,
            palette=custom_palette,
        )
        axes[0].set_title("Sentiment Analysis vs Rating Score")

        # Second subplot
        sns.countplot(x="rating_score", hue="label3", data=data, ax=axes[1])
        axes[1].set_title(
            "Positive/Neutral/Negative Sentiment Analysis Distribution per Rating Score"
        )
        axes[1].legend(title="Sentiment", loc="upper left")

        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)

elif page == "Text Classification":

    st.header("Text Classification")
    
    if st.checkbox("Review Last Month Only"):
        # Last Months Results
        # Convert the 'time' column to datetime
        data['time'] = pd.to_datetime(data['time'])

        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year

        # Filter data for the current month and year
        current_month_data = data[(data['time'].dt.month == current_month) & (data['time'].dt.year == current_year)]

        # Now, proceed with your original code but use the filtered DataFrame
        class_list = list(current_month_data['classif'].unique())
        selected_rating = st.selectbox("Viewing Patient Feedback by Classification (Current Month Only):", class_list)

        filtered_class = current_month_data[current_month_data["classif"] == selected_rating]
        st.subheader(f'{selected_rating.capitalize()} ({str(filtered_class.shape[0])})')
        for text in filtered_class['free_text']:  # Assuming 'free_text' is the column with the text you want to display
            if str(text) != 'nan':
                st.write('- ' + str(text))
        
    else:
        class_list = list(data['classif'].unique())
        selected_rating = st.selectbox("Viewing Patient Feedback by Classification (All Reviews):", class_list)
        
        filtered_class = data[data["classif"] == selected_rating]
        st.subheader(f'{selected_rating.capitalize()} ({str(filtered_class.shape[0])})')
        for text in filtered_class['free_text']:  # Assuming 'Review' is the column with the text you want to display
            if str(text) != 'nan':
                st.write('- ' + str(text))

elif page == "Word Cloud":
    st.header("Word Cloud")
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
