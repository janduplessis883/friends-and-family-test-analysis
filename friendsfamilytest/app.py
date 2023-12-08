import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime


def load_data():
    df = pd.read_csv("friendsfamilytest/data/data.csv")
    df['time'] = pd.to_datetime(df['time'])
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
        "Feedback Word Cloud",
        "View Dataframe",
        "About",
    ],
)


# Display content based on the selected page
if page == "Monthly Rating & Count":
    st.image(
        "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
        use_column_width=True,
    )
    st.header("Average Monthly Rating")
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
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    st.pyplot(fig)
    
    
    st.markdown(
        '''The Friends and Family Test (FFT) is a feedback tool used in the healthcare sector, particularly in the UK's National Health Service (NHS), to help measure patient satisfaction with services. It allows patients to provide feedback on their experience with a particular service, including General Practitioner (GP) surgeries. The test is straightforward, usually asking whether the patient would recommend the service to friends and family if needed. 
        \nPatients can typically respond with options like "extremely likely," "likely," "neither likely nor unlikely," "unlikely," "extremely unlikely," and "don't know." 
        \nThis visualization aids in quickly assessing the performance and patient satisfaction over the months in question.''')

    st.markdown("---")
    # Create two columns
    col1, col2 = st.columns(2)

    # Use the columns
    with col1:
        st.image(
            "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/arrow.png?raw=true",
            use_column_width=True,
        )

    with col2:
        st.markdown(f'# {data.shape[0]}')
        st.write(f'Total **Feedback** since Aug 23')
    
    st.markdown("---")

    # Resample and count the entries per day
    monthly_count = data_time.resample("M").size()
    # Reset index to convert Series to DataFrame
    monthly_count = monthly_count.reset_index(name="entry_count")
    st.header("Monthly Responses")
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
    ax.xaxis.grid(False)

    # Removing top, left and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Annotating each bar with the height (number of entries)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')

    # Show the plot in Streamlit
    st.pyplot(fig)
    st.write(
        """A "FFT (Friends and Family Test) Count per Month" plot is a visual representation used to display the number of responses received for the FFT in a GP surgery over a series of months. This type of plot is particularly useful for understanding patient engagement and the volume of feedback over time. """
    )


elif page == "Rating & Sentiment Analysis Correlation":
    st.header("Rating & Sentiment Analysis Correlation")

    if st.checkbox("Current Month Only"):
        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]

        plt.figure(figsize=(10, 6))  # Adjusting the figure size
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

        # Removing the left, top, and right spines
        scatter_plot.spines['left'].set_visible(False)
        scatter_plot.spines['top'].set_visible(False)
        scatter_plot.spines['right'].set_visible(False)

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
        
        # Removing the left, top, and right spines
        scatter_plot.spines['left'].set_visible(False)
        scatter_plot.spines['top'].set_visible(False)
        scatter_plot.spines['right'].set_visible(False)

        # Display the plot in Streamlit
        st.pyplot(plt)
        st.subheader("Viewing All Data")
        st.write(
            """The plot maps 'rating_score' along the x-axis and 'sentiment_score' along the y-axis. Points on the scatter plot are color-coded to represent three categories of sentiment: positive (blue), neutral (orange), and negative (green). Most of the data points appear to be concentrated at the higher end of the rating scale (closer to 5.0), suggesting a large number of positive sentiment scores. The spread and density of points suggest that higher rating scores correlate with more positive sentiment."""
        )

elif page == "Feedback Classification":
    st.header("Feedback Classification")

    if st.checkbox("Current Month Only"):
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
        
        # Calculate value counts
        label_counts = current_month_data['classif'].value_counts(ascending=False) # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ['Feedback Classification', 'Counts']

        # Define the palette conditionally based on the category names
        palette = ['#ddec9c' if (label == 'Overall Patient Satisfaction' or label == 'No Improvment Suggestion') else '#4088a9' for label in label_counts_df['Feedback Classification']]

        # Create a Seaborn bar plot
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x='Counts', y='Feedback Classification', data=label_counts_df, color="#2d668f")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        
        # Adding titles and labels for clarity
        plt.title('Counts of Feedback Classification')
        plt.xlabel('Counts')
        plt.ylabel('')

        # Streamlit function to display matplotlib figures
        st.pyplot(plt)

        st.subheader("View Patient Feedback")
        # Now, proceed with your original code but use the filtered DataFrame
        class_list = list(current_month_data["classif"].unique()) 
        selected_ratings = st.multiselect(
            "Select a Feedback Categories:",
            class_list,
        )

        # Filter based on the selected classifications
        filtered_classes = current_month_data[current_month_data["classif"].isin(selected_ratings)]

        # If there are no selected ratings, don't display anything
        if not selected_ratings:
            st.warning("Please select at least one classification.")
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[filtered_classes["classif"] == rating]
                st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
                for text in specific_class["free_text"]:  # Assuming 'free_text' is the column with the text you want to display
                    if str(text) != "nan":
                        st.write("- " + str(text))


    else:
        
        # Calculate value counts
        label_counts = data['classif'].value_counts(ascending=False) # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ['Feedback Classification', 'Counts']

        # Define the palette conditionally based on the category names
        palette = ['#ddec9c' if (label == 'Overall Patient Satisfaction' or label == 'No Improvment Suggestion') else '#4088a9' for label in label_counts_df['Feedback Classification']]

        # Create a Seaborn bar plot
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x='Counts', y='Feedback Classification', data=label_counts_df, color="#2d668f")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        # Adding titles and labels for clarity
        plt.title('Counts of Feedback Classification')
        plt.xlabel('Counts')
        plt.ylabel('')

        # Streamlit function to display matplotlib figures
        st.pyplot(plt)
        st.subheader("View Patient Feedback")
        class_list = list(data["classif"].unique())
        selected_ratings = st.multiselect(
            "Select Feedback Categories:", 
            class_list
        )

        # Filter the data based on the selected classifications
        filtered_classes = data[data["classif"].isin(selected_ratings)]

        if not selected_ratings:
            st.warning("Please select at least one classification.")
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[filtered_classes["classif"] == rating]
                st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
                for text in specific_class["free_text"]:  # Assuming 'free_text' is the column with the text you want to display
                    if str(text) != "nan":
                        st.write("- " + str(text))


elif page == "Feedback Word Cloud":
    st.header("Feedback Word Cloud")
    if st.checkbox("Current Month Only"):

        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]
        
        st.subheader("Feedback")
        text = " ".join(current_month_data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
        st.markdown("---")
        st.subheader("Improvement")
        text2 = " ".join(current_month_data["do_better"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Reds").generate(text2)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.subheader("Feedback")
        text = " ".join(data["free_text"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Blues").generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
        st.markdown("---")
        st.subheader("Improvement")
        text2 = " ".join(data["do_better"].dropna())
        wordcloud = WordCloud(background_color="white", colormap="Reds").generate(text2)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
        
elif page == "Improvement Suggestions":
    st.header("Improvement Suggestions")
    
    exclude_list = ['fine', 'no', 'nan', 'not', 'ok', 'nothing', 'anything', 'okay', 'nathing', 'good', 'excellent', 'happy', 'professionally', 'professional', 'amazing', 'thanks', 'satisfied', 'yes', 'na', 'thank']
    
    if st.checkbox("Current Month Only"):
        
        data["time"] = pd.to_datetime(data["time"])
        # Get the current month and year
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Filter data for the current month and year
        current_month_data = data[
            (data["time"].dt.month == current_month)
            & (data["time"].dt.year == current_year)
        ]
    
        # Calculate value counts
        label_counts = current_month_data['improvement_labels'].value_counts(ascending=False) # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ['Improvement Labels', 'Counts']

        # Define the palette conditionally based on the category names
        palette = ['#5c6853' if (label == 'Overall Patient Satisfaction' or label == 'No Improvment Suggestion') else '#168aad' for label in label_counts_df['Improvement Labels']]

        # Create a Seaborn bar plot
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x='Counts', y='Improvement Labels', data=label_counts_df, palette=palette)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        # Adding titles and labels for clarity
        plt.title('Counts of Improvement Categories')
        plt.xlabel('Counts')
        plt.ylabel('')

        # Streamlit function to display matplotlib figures
        st.pyplot(plt)
        
        st.subheader("View Patient Improvement Suggestions")
        
        improvement_list = [label for label in label_counts_df['Improvement Labels']]

        selected_ratings = st.multiselect(
            "Select Categories:", 
            improvement_list
        )

        # Filter the data based on the selected classifications
        # Ensure you're filtering current_month_data, not data
        if selected_ratings:
            filtered_classes = current_month_data[current_month_data["improvement_labels"].isin(selected_ratings)]
            for rating in selected_ratings:
                specific_class = filtered_classes[filtered_classes["improvement_labels"] == rating]
                st.subheader(f"{rating.capitalize()} ({str(specific_class.shape[0])})")
                for text in specific_class["do_better"]:  # Assuming 'do_better' is the column with the text you want to display
                    st.write("- " + str(text))
        else:
            st.warning("Please select at least one classification.")
        
   
    else:
        # Calculate value counts
        label_counts = data['improvement_labels'].value_counts(ascending=False) # Use ascending=True to match the order in your image

        # Convert the Series to a DataFrame
        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ['Improvement Labels', 'Counts']

        # Define the palette conditionally based on the category names
        palette = ['#5c6853' if (label == 'Overall Patient Satisfaction' or label == 'No Improvment Suggestion') else '#168aad' for label in label_counts_df['Improvement Labels']]

        # Create a Seaborn bar plot
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x='Counts', y='Improvement Labels', data=label_counts_df, palette=palette)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
        ax.yaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        # Adding titles and labels for clarity
        plt.title('Counts of Improvement Catergories')
        plt.xlabel('Counts')
        plt.ylabel('')

        # Streamlit function to display matplotlib figures
        st.pyplot(plt)
        st.subheader("View Patient Improvement Suggestions")
        improvement_list = [label for label in label_counts_df['Improvement Labels']]
    
        selected_ratings = st.multiselect(
            "Select Categories:", 
            improvement_list
        )

        # Filter the data based on the selected classifications
        filtered_classes = data[data["improvement_labels"].isin(selected_ratings)]

        if not selected_ratings:
            st.warning("Please select at least one classification.")
        else:
            for rating in selected_ratings:
                specific_class = filtered_classes[filtered_classes["improvement_labels"] == rating]
                st.subheader(f"{str(rating).capitalize()} ({str(specific_class.shape[0])})")
                for text in specific_class["do_better"]:  # Assuming 'free_text' is the column with the text you want to display
                    st.write("- " + str(text))
 
                
elif page == "View Dataframe":
    st.header("Dataframe")
    # Create a slider and get the number of rows to display
    num_rows_to_display = st.slider('Select number of rows to display', 5, len(data), 10)

    # Display the specified number of rows from the DataFrame
    st.dataframe(data.tail(num_rows_to_display))

    
    
elif page == "About":
    st.image(
    "https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftest2.png?raw=true",
    use_column_width=True,
    )
    st.header("About")
    st.subheader("Friends & Family Test (FFT) Dashboard")
    st.markdown(
        """Welcome to our new dashboard, aimed at enhancing how healthcare providers understand and use patient feedback. This tool focuses on the Friends and Family Test (FFT), which is essential for collecting patients' views on healthcare services. Our approach uses advanced text classification and sentiment analysis to organize and interpret this feedback more effectively.

Here's the core idea: Instead of just counting responses, the dashboard analyzes the sentiments behind them—whether positive, negative, or neutral. It assigns a detailed score to each piece of feedback, allowing for a more nuanced understanding of patient satisfaction. This method helps identify specific areas needing improvement and those that are performing well, based on real patient experiences.

For healthcare providers, this tool offers a more insightful way to look at patient feedback. It doesn’t just provide data; it offers a clearer picture of how patients feel about their care. This can help highlight excellence in services and pinpoint areas for potential improvements.

The data we use comes from a GP surgery in West London, showing how this tool can be applied in a real healthcare setting.

We employ several machine learning techniques for analysis:

1. **Sentiment Analysis:** Using Huggingface's 'cardiffnlp/twitter-roberta-base-sentiment-latest' model, we determine the emotional tone of the feedback.
2. **Text Classification of Patient Feedback:** To categorize feedback into different emotional themes, we use the 'SamLowe/roberta-base-go_emotions' model from Huggingface.
3. **Zero-shot Classification of Patient Improvement Suggestions:** The 'facebook/bart-large-mnli' model helps us identify and classify suggestions for improving patient care, even when the model hasn’t been specifically trained on healthcare data.

This system is available through a Streamlit app developed by janduplessis883, making it easy for healthcare professionals to use.

So, join us in using this tool to better understand and respond to patient feedback, aiming to improve healthcare delivery based on what patients actually say and feel.""")

    st.markdown('---')
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
        st.markdown("**Text Classification** and **Sentiment Analysis** by Huggingface.co")

