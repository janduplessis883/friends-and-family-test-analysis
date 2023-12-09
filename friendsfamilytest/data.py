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
from friendsfamilytest.auto_git.git_push import *

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
        yield data[column_name][
            i : i + batch_size
        ], i  # Yield the batch and the starting index


def improvement_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    # Labels
    improvement_labels_list = [
        "Reception Services",
        "Ambiance of Facility",
        "Modernization and Upgrades",
        "Nursing Quality",
        "Waiting Times",
        "Referral Process",
        "Staff Knowledge",
        "Staffing Levels",
        "Facility Accessibility",
        "Poor Communication",
        "Online Services",
        "Patient Safety",
        "Weekend Service Availability",
        "Location Convenience",
        "Telephone Service",
        "Hydration and Catering Facilities",
        "Website Usability",
        "Medical Equipment",
        "After-Hours Service",
        "Staff Training and Development",
        "Diagnostic Services",
        "Prescription Process",
        "Poor Quality of Medical Advice",
        "Overall Patient Satisfaction",
        "Appointment System Efficiency",
        "Efficiency in Test Result Delivery",
        "Patient Participation Group",
        "Mental Health Services",
        "Telehealth and Technology Use",
        "Social Prescribing Services",
        "Chronic Disease Managment",
        "No Improvment Suggestion",
        "Displeased with Medical Consultation",
        "More Face to Face Consultations",
        "Home Visits",
    ]

    # Initialize the list to store labels
    improvement_labels = [""] * len(data)  # Pre-fill with empty strings

    # Iterate over batches
    for batch, start_index in batch_generator(data, "do_better", batch_size):
        # Filter out empty or whitespace-only sentences
        valid_sentences = [
            (sentence, idx)
            for idx, sentence in enumerate(batch)
            if sentence and not sentence.isspace()
        ]
        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        # Classify the batch
        if sentences:
            model_outputs = classifier(
                list(sentences), improvement_labels_list, device="cpu"
            )
            # Assign labels to corresponding indices
            for output, idx in zip(model_outputs, valid_indices):
                improvement_labels[start_index + idx] = output["labels"][0]
                print(
                    f"{Fore.GREEN}Batch processed: {start_index + idx + 1}/{len(data)}"
                )

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
    print(f"{Fore.RED}[+] Loading Pre-processed data (processed_data from data.csv)")
    processed_data = pd.read_csv(f"{DATA_PATH}/data.csv")
    processed_data["time"] = pd.to_datetime(processed_data["time"], dayfirst=True)
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
    print(
        f"{Fore.YELLOW}[i] 💾 Concat dataframes and save combined_data to '/data/data.csv'"
    )
    # Concatenate the DataFrames one below the other
    combined_data = pd.concat([processed_data, data], ignore_index=True)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", index=False)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Call Auto Git Push Master
    do_git_push()
