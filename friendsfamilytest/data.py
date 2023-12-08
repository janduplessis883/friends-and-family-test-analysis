import os
import time
import pandas as pd
from transformers import pipeline
from sheethelper import SheetHelper
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
import warnings
import subprocess
from datetime import datetime

from friendsfamilytest.params import *
from friendsfamilytest.utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
secret_path = os.getenv("SECRET_PATH")
init(autoreset=True)


def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1K2d32XmZQMdGLslNzv2ZZoUquARl6yiKRT5SjUkTtIY/edit#gid=1323317089",
        sheet_id=0,
    )
    data = sh.gsheet_to_df()
    data.columns = ["time", "rating", "free_text", "do_better"]
    data["time"] = pd.to_datetime(data["time"], format="%d/%m/%Y %H:%M:%S")

    data["do_better"] = data["do_better"].apply(clean_and_replace)
    data["free_text"] = data["free_text"].apply(clean_and_replace)
    return data

def update_datetime_format(df, column_name):
    """
    Update the format of a datetime column in a DataFrame.

    :param df: DataFrame containing the datetime column.
    :param column_name: Name of the column to be formatted.
    :return: DataFrame with the updated datetime format.
    """

    # First, ensure the column is in datetime forma
    df[column_name] = pd.to_datetime(df[column_name])

    return df

# Example usage:
# Assuming you have a DataFrame 'data' with a datetime column named 'time'
# data = update_datetime_format(data, 'time')

def text_classification(data):
    # Initialize classifier
    classifier = pipeline(
        task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
    )

    # Initialize lists to store labels and scores
    classif = []
    classif_scores = []

    # Iterate over DataFrame rows and classify text
    for _, row in data.iterrows():
        sentence = row["free_text"]
        model_outputs = classifier(sentence)
        classif.append(model_outputs[0][0]["label"])
        classif_scores.append(model_outputs[0][0]["score"])

    # Add labels and scores as new columns
    data["classif"] = classif
    data["classif_scores"] = classif_scores

    return data


def sentiment_analysis(data):
    sentiment_task = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # Initialize lists to store labels and scores
    sentiment = []
    sentiment_score = []

    # Iterate over DataFrame rows and classify text
    for _, row in data.iterrows():
        sentence = row["free_text"]
        model_outputs = sentiment_task(sentence)
        sentiment.append(model_outputs[0]["label"])
        sentiment_score.append(model_outputs[0]["score"])

    # Add labels and scores as new columns
    data["sentiment"] = sentiment
    data["sentiment_score"] = sentiment_score

    return data


def summarization(data):
    summ = pipeline("summarization", model="Falconsai/text_summarization")

    # Initialize lists to store labels and scores
    summ_list = []

    # Iterate over DataFrame rows and classify text
    for _, row in data.iterrows():
        sentence = row["free_text"]
        if sentence != "":
            sentence_length = len(sentence.split())
            if sentence_length > 10:
                model_outputs = summ(
                    sentence,
                    max_length=int(sentence_length - (sentence_length / 3)),
                    min_length=1,
                    do_sample=False,
                )
                summ_list.append(model_outputs[0]["summary_text"])
                print(f"{Fore.RED}{sentence}")
                print(model_outputs[0]["summary_text"])
            else:
                summ_list.append("")
        else:
            summ_list.append("")
    data["free_text_summary"] = summ_list

    return data

# Zer0-shot classification - do_better column
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

def batch_generator(data, column_name, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[column_name][i:i + batch_size], i  # Yield the batch and the starting index

def improvement_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to('cpu')
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    # Labels
    improvement_labels_list = [
        "Reception Service",
        "Ambiance of Facility",
        "Modernization and Upgrades",
        "Nursing Quality",
        "Waiting Times",
        "Referral Process",
        "Staff Knowledge",
        "Staffing Levels",
        "Facility Accessibility",
        "Quality of Results",
        "Communication Effectiveness",
        "Online Services",
        "Patient Safety",
        "Weekend Service Availability",
        "Patient Contentment",
        "Location Convenience",
        "Telephone Service",
        "Physiotherapy Services",
        "Hydration Facilities",
        "Seating Comfort",
        "Website Usability",
        "Educational Resources",
        "Medical Equipment",
        "After-Hours Service",
        "Staff Training",
        "Diagnostic Services",
        "Prescription Process",
        "Operational Efficiency",
        "Medical Advice Quality",
        "Patient Experience",
        "Overall Satisfaction",
        "Appointment Availability",
    ]




    # Initialize the list to store labels
    improvement_labels = [''] * len(data)  # Pre-fill with empty strings

    # Iterate over batches
    for batch, start_index in batch_generator(data, "do_better", batch_size):
        # Filter out empty or whitespace-only sentences
        valid_sentences = [(sentence, idx) for idx, sentence in enumerate(batch) if sentence and not sentence.isspace()]
        sentences, valid_indices = zip(*valid_sentences) if valid_sentences else ([], [])
        
        # Classify the batch
        if sentences:
            model_outputs = classifier(list(sentences), improvement_labels_list, device='cpu')
            # Assign labels to corresponding indices
            for output, idx in zip(model_outputs, valid_indices):
                improvement_labels[start_index + idx] = output["labels"][0]
                print(f"{Fore.GREEN}Batch processed: {start_index + idx + 1}/{len(data)}")

    # Add labels as a new column
    data["improvement_labels"] = improvement_labels
    
    return data

# Example usage
# Assuming 'data' is your DataFrame and 'do_better' is the column with sentences
# data = improvement_classification(data, batch_size=16)



def add_rating_score(data):
    # Mapping dictionary
    rating_map = {
        "Extremely likely": 5,
        "Likely": 4,
        "Neither likely nor unlikely": 3,
        "Unlikely": 2,
        "Extremely unlikely": 1,
    }

    # Apply the mapping to the 'rating' column
    data["rating_score"] = data["rating"].map(rating_map)
    return data


if __name__ == "__main__":
    print(f"{Fore.WHITE}{Back.BLACK}[*] Parsing Friends & Family Test Data")

    start_time = time.time()
    print(f"{Fore.RED}[+] Google Sheet Loading (raw data)")
    raw_data = load_google_sheet()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    print(f"{Fore.RED}[+] Loading Pre-processed data (raw data from data.csv)")
    processed_data = pd.read_csv(f"{DATA_PATH}/data.csv")
    processed_data['time'] = pd.to_datetime(processed_data['time'], format="%d/%m/%Y %H:%M")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    print(f"{Fore.RED}[+] Identify new data to process")
    data = raw_data[~raw_data.index.isin(processed_data.index)]
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    print(f"{Fore.BLUE}[+] Rating score added")
    data = add_rating_score(data)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print(f"{Fore.BLUE}[+] Text Classification")
    data = text_classification(data)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print(f"{Fore.BLUE}[+] Sentiment Analysis")
    data = sentiment_analysis(data)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print(f"{Fore.BLUE}[+] Improvement Classification - do_better column")
    data = improvement_classification(data, batch_size=16)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print(f"{Fore.YELLOW}[i] 💾 Append new data to '/data/data.csv'")
    # Check if the CSV file ends with a newline and add one if not
    with open(f'{DATA_PATH}/data.csv', 'a+', newline='') as f:
        f.seek(0, 2)  # Move to the end of the file
        if f.tell() == 0 or f.read(1) != '\n':
            f.write('\n')
            
    data = update_datetime_format(data, 'time')
    data.to_csv(f'{DATA_PATH}/data.csv', mode='a', header=False, index=False)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print(f"{Fore.WHITE}{Back.BLACK}[>] Git: Push to GitHub Repo")

    repo_path = LOCAL_GIT_REPO
    remote = "origin"
    branch = "master"
    os.chdir(repo_path)

    subprocess.run(["git", "add", "."])
    print(f"{Fore.RED}[+] Git: commit")
    
    # Get the current date and time
    current_timestamp = datetime.now()
    # Format the timestamp to include date, hour, and minute
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M")

    message = f"Automated commit from Python script - {formatted_timestamp}"
    subprocess.run(["git", "commit", "-m", message])
    print(f"{Fore.RED}[+] Git: push to remote {branch}")
    subprocess.run(["git", "push", remote, branch])
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    print(f"{Fore.YELLOW}{Back.GREEN}[i] ✅ data.csv push to GitHub successful")