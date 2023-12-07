import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime


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
        "Monthly Rating & Count",
        "Rating & Sentiment Analysis Correlation",
        "Feedback Classification",
        "Feedback Word Cloud",
        "Improvement Opportunities",
        "About",
    ],
)


# Display content based on the selected page
if page == "Monthly Rating & Count":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )
    st.header('Monthly Rating & Count')
    # Plot monthly averages

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
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    st.pyplot(fig)

    st.subheader("Average Monthly Rating")
    st.write(
        '''The Friends and Family Test (FFT) is a feedback tool used in the healthcare sector, particularly in the UK's National Health Service (NHS), to help measure patient satisfaction with services. It allows patients to provide feedback on their experience with a particular service, including General Practitioner (GP) surgeries. The test is straightforward, usually asking whether the patient would recommend the service to friends and family if needed. Patients can typically respond with options like "extremely likely," "likely," "neither likely nor unlikely," "unlikely," "extremely unlikely," and "don't know."'''
    )

    st.markdown("---")

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
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    # Show the plot in Streamlit
    st.pyplot(fig)

    st.subheader("No of Reviews per Month")
    st.write(
        """A "FFT (Friends and Family Test) Count per Month" plot is a visual representation used to display the number of responses received for the FFT in a GP surgery over a series of months. This type of plot is particularly useful for understanding patient engagement and the volume of feedback over time. """
    )


elif page == "Rating & Sentiment Analysis Correlation":
    st.header("Rating & Sentiment Analysis Correlation")

    if st.checkbox("Review Last Month Only"):
        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]

        plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
        scatter_plot = sns.scatterplot(
            data=current_month_data,
            x="rating_score",
            y="sentiment_score",
            hue="sentiment",
            s=55,
        )

        # Setting x-axis ticks to 1, 2, 3, 4, 5
        scatter_plot.set_xticks([0.5, 1, 2, 3, 4, 5])
        plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

        # Display the plot in Streamlit
        st.pyplot(plt)
        st.subheader("Viewing Last Month's Data Only")
        st.write(
            """The plot maps 'rating_score' along the x-axis and 'sentiment_score' along the y-axis. Points on the scatter plot are color-coded to represent three categories of sentiment: positive (blue), neutral (orange), and negative (green). Most of the data points appear to be concentrated at the higher end of the rating scale (closer to 5.0), suggesting a large number of positive sentiment scores. The spread and density of points suggest that higher rating scores correlate with more positive sentiment."""
        )

    else:
        plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
        scatter_plot = sns.scatterplot(
            data=data,
            x="rating_score",
            y="sentiment_score",
            hue="sentiment",
            s=55,
        )

        # Setting x-axis ticks to 1, 2, 3, 4, 5
        scatter_plot.set_xticks([0.5, 1, 2, 3, 4, 5])
        plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

        # Display the plot in Streamlit
        st.pyplot(plt)
        st.subheader("Viewing All Data")
        st.write(
            """The plot maps 'rating_score' along the x-axis and 'sentiment_score' along the y-axis. Points on the scatter plot are color-coded to represent three categories of sentiment: positive (blue), neutral (orange), and negative (green). Most of the data points appear to be concentrated at the higher end of the rating scale (closer to 5.0), suggesting a large number of positive sentiment scores. The spread and density of points suggest that higher rating scores correlate with more positive sentiment."""
        )

elif page == "Feedback Classification":
    st.header("Feedback Classification")

    if st.checkbox("Review Last Month Only"):
        # Last Months Results
        # Convert the 'time' column to datetime
        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]

        # Now, proceed with your original code but use the filtered DataFrame
        class_list = list(current_month_data["classif"].unique())
        selected_rating = st.selectbox(
            "Viewing Patient Feedback by Classification (❗️Current Month Only):",
            class_list,
        )

        filtered_class = current_month_data[
            current_month_data["classif"] == selected_rating
        ]
        st.subheader(f"{selected_rating.capitalize()} ({str(filtered_class.shape[0])})")
        for text in filtered_class[
            "free_text"
        ]:  # Assuming 'free_text' is the column with the text you want to display
            if str(text) != "nan":
                st.write("- " + str(text))

    else:
        class_list = list(data["classif"].unique())
        selected_rating = st.selectbox(
            "Viewing Patient Feedback by Classification ( ✅ All Reviews):", class_list
        )

        filtered_class = data[data["classif"] == selected_rating]
        st.subheader(f"{selected_rating.capitalize()} ({str(filtered_class.shape[0])})")
        for text in filtered_class[
            "free_text"
        ]:  # Assuming 'Review' is the column with the text you want to display
            if str(text) != "nan":
                st.write("- " + str(text))

elif page == "Feedback Word Cloud":
    st.header("Feedback Word Cloud")
    if st.checkbox("Display Last Month Only"):

        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]
        text = " ".join(current_month_data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        text = " ".join(data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
        
elif page == "Improvement Opportunities":
    st.header("Improvement Opportunities")
    exclude_list = ['fine', 'no', 'nan', 'not', 'ok', 'nothing', 'anything', 'okay', 'nathing', 'good', 'excellent', 'happy', 'professionally', 'professional', 'amazing', 'thanks', 'satisfied', 'yes', 'na', 'thank']
    
    if st.checkbox("Display Last Month Only"):
        st.subheader("Last Month's Suggestions")
        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]
    
        for text in current_month_data['do_better']:
            words = str(text).lower().split()  # Split the text into words and convert to lowercase
            if not any(word in exclude_list for word in words):
                st.write("- " + str(text))
                
    else:
        st.subheader('All Suggestions')
        for text in data['do_better']:
            words = str(text).lower().split()  # Split the text into words and convert to lowercase
            if not any(word in exclude_list for word in words):
                st.write("- " + str(text))

elif page == "About":
    st.header("About this App")
    st.subheader("Welcome to the Friends & Family Test (FFT) Dashboard")
    st.markdown(
        """The FFT is a cornerstone of patient feedback, offering invaluable insights into the quality of healthcare services through the eyes of those who matter most — the patients. Our dashboard harnesses the power of advanced text classification and sentiment analysis to seamlessly sift through and categorize a wealth of patient feedback into distinct themes.

\nWhat does this mean for healthcare? It translates patient voices into actionable data. By scrutinizing the sentiment behind each response, whether positive, negative, or neutral, we assign a nuanced polarity score that goes beyond mere numbers. It's a deep dive into patient satisfaction, providing a clear view of performance and pinpointing specific areas that need attention or deserve applause.

\nHealthcare providers gain a comprehensive understanding of patient experiences, allowing them to celebrate excellence in care and address areas needing refinement. It's more than a feedback loop; it's a pathway to data-driven enhancements in patient care.

\nThis dashboard draws on data from a dedicated GP surgery in West London, reflecting real-world applications of patient-centered healthcare analysis.

\nWelcome aboard — let's navigate the nuances of patient feedback together and steer towards exceptional healthcare delivery.
             \n Streamlit App by [janduplessis883](https://github.com/janduplessis883)"""
    )
    st.markdown("---")
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/about.png?raw=true",
        width=200,
    )
