import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

@st.cache_data  # This decorator will help you cache the data
def load_data():
    df = pd.read_csv('friendsfamilytest/data/data.csv')
    return df

data = load_data()

def load_timedata():
    df = pd.read_csv('friendsfamilytest/data/data.csv')
    df['time'] = pd.to_datetime(df['time']) 
    df.set_index('time', inplace=True)
    return df

# Calculate monthly averages
data_time = load_timedata()

monthly_avg = data_time['rating_score'].resample('M').mean()
monthly_avg_df = monthly_avg.reset_index()
monthly_avg_df.columns = ['Month', 'Average Rating']

st.sidebar.title('Menu') 
page = st.sidebar.selectbox('Choose an option', ['Monthly Rating', 'Monthly Count', 'Sentiment Analysis Histplot', 'Rating & Sentiment Correlation', 'Text Classification', 'Word Cloud', 'Show Raw Data', 'About'])
st.sidebar.slider('View data for last n months?', min_value=1, max_value=12, value=1)

# Display content based on the selected page
if page == 'Monthly Rating':
    st.title('Friends & Family Test Monthly Rating')
    st.write('The monthly rating by rating score.')
    
    # Plot monthly averages
    st.subheader('Monthly Average Rating')

    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x='Month', y='Average Rating', data=monthly_avg_df, color='#a54b49', linewidth=3)
    plt.title('Monthly Average Rating')
    plt.xticks(rotation=45)

    # Annotate each point with its value
    for index, row in monthly_avg_df.iterrows():
        ax.annotate(f"{row['Average Rating']:.2f}", 
                    (row['Month'], row['Average Rating']),
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')  # horizontal alignment can be left, right or center

    # Display the plot in Streamlit
    st.pyplot(fig)
    
elif page == 'Monthly Count':
    st.title('Monthly Count')
    st.write('Showing the number of FF Test responses received per month.')
    
    # Resample and count the entries per day
    monthly_count = data_time.resample('M').size()
    
    # Reset index to convert Series to DataFrame
    monthly_count = monthly_count.reset_index(name='entry_count')
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=monthly_count, x='time', y='entry_count', color='#a54b49')
    
    # Customizing x-axis labels
    n = len(monthly_count['time'])
    tick_frequency = n // 4  # Adjust as needed
    plt.xticks(ticks=range(0, n, tick_frequency), labels=[monthly_count['time'].iloc[i].strftime('%Y-%m-%d') for i in range(0, n, tick_frequency)], rotation=45)
    plt.title('Friend & Family Test Responses per Month')
    
    # Show the plot in Streamlit
    st.pyplot(fig)
    
elif page == 'Sentiment Analysis Histplot':
    st.title('Sentiment Analysis Histplot')
    st.write('You are viewing the content of Page 2.')
    
    # Filter the data for 'Negative' and 'Positive' values in the 'score3' column
    data_positive = data[data['label3'] == 'positive']['score3']
    data_negative = data[data['label3'] == 'negative']['score3']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the histograms
    sns.histplot(data_positive, color='#69a443', bins=20, kde=True, label='Positive', ax=ax)
    sns.histplot(data_negative, color='#a54b49', bins=20, kde=True, label='Negative', ax=ax)

    # Add title and labels
    plt.title('Sentiment Analysis of FF Test Responses')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot
    st.pyplot(fig)
    
elif page == 'Rating & Sentiment Correlation':
    st.title('Rating & Sentiment Correlation')
    st.write('You are viewing the content of Page 2.')

    st.subheader('Correlation between score3 and rating_score')
    correlation = data['score3'].corr(data['rating_score'])
    st.write(f'Correlation: {correlation}')
    
    # Scatter Plot
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.scatter(data['score3'], data['rating_score'], color='#a54b49', alpha=0.7, marker="x")
    ax1.set_title('Scatter Plot of Sentiment Analysis Score vs Rating Score')
    ax1.set_xlabel('Sentiment Analysis Score')
    ax1.set_ylabel('Rating Score')
    st.pyplot(fig1)
    
elif page == 'Text Classification':
    st.title('Text Classification')
    st.write('You are viewing the content of Page 2.')
    
   # Prepare the data
    label_counts = data['label1'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']

    # Sort the dataframe by count
    label_counts = label_counts.sort_values(by='count', ascending=False)

    # Display DataFrame with index set to 3
    st.write('Text Classification Frequency')

    # Plot using seaborn
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=label_counts, y='label', x='count', color="#43597d")
    
    # Display the plot in Streamlit
    st.pyplot(fig3)
    
elif page == 'Word Cloud':
    st.title('Word Cloud')
    st.write('You are viewing the content of Page 2.')
    
    st.subheader('Word Cloud')
    text = ' '.join(data['free_text'].dropna())
    wordcloud = WordCloud(background_color='white', colormap='cividis').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
    
elif page == 'Show Raw Data':
    st.title('Show Raw Data')
    st.write('You are viewing the content of Page 2.')
    
    # Create a slider for selecting the number of rows to display
    # Create a slider for selecting the number of rows to display
    num_rows = st.slider('Select number of rows to display:', min_value=5, max_value=100)

    # Create a dropdown for selecting the rating_score to filter by
    selected_rating = st.selectbox('Select rating score to filter by:', [1, 2, 3, 4, 5])

    # Filter the DataFrame based on the selected rating_score
    filtered_data = data[data['rating_score'] == selected_rating]

    # Use the slider's value to display that many rows from the tail of the filtered DataFrame
    if num_rows:
        st.subheader(f'Displaying last {num_rows} rows of raw data with rating score {selected_rating}')
        st.write(filtered_data.tail(num_rows))
    
elif page == 'About':
    st.title('About')
    st.write('You are viewing the content of Page 2.')