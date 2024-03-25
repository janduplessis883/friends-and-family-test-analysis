import os
import pandas as pd
from transformers import pipeline
from sheethelper import SheetHelper
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import torch

from friendsfamilytest.params import *
from friendsfamilytest.utils import *
from friendsfamilytest.auto_git.git_merge import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")
secret_path = os.getenv("SECRET_PATH")
from sheethelper import *
import cronitor

cronitor.api_key = os.getenv("CRONITOR_API_KEY")
from loguru import logger

logger.add("log/debug.log", rotation="500 KB")


@time_it
def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1K2d32XmZQMdGLslNzv2ZZoUquARl6yiKRT5SjUkTtIY/edit#gid=1323317089",
        sheet_id=0,
    )
    data = sh.gsheet_to_df()
    data.columns = ["time", "rating", "free_text", "do_better", "surgery"]
    data["time"] = pd.to_datetime(data["time"], format="%d/%m/%Y %H:%M:%S")
    data.sort_values(by="time", inplace=True)
    return data


@time_it
def word_count(df):
    df["free_text_len"] = df["free_text"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else np.nan
    )

    df["do_better_len"] = df["do_better"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else np.nan
    )

    return df


@time_it
def clean_text(df):
    df["do_better"] = df["do_better"].apply(clean_and_replace)
    df["free_text"] = df["free_text"].apply(clean_and_replace)

    return df


@time_it
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


@time_it
def check_column_length(dataframe, column_name, word_count_length):
    # Iterate over each entry in the specified column
    for index, entry in enumerate(dataframe[column_name]):
        # Count the number of words in the entry
        word_count = len(str(entry).split())

        # Check if the word count is less than the specified limit
        if word_count < word_count_length:
            # Replace with NaN if the condition is met
            dataframe.at[index, column_name] = np.nan

    return dataframe


sentiment_task = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


@time_it
def sentiment_analysis(data):

    # Initialize lists to store labels and scores
    sentiment = []
    sentiment_score = []

    # Iterate over DataFrame rows and classify text
    for index, row in data.iterrows():
        print(index)
        freetext = row["free_text"]
        dobetter = row['do_better']
        sentence = str(freetext) + ' ' + str(dobetter)
        sentence = sentence[:513]
        if pd.isna(sentence) or sentence == "":
            sentiment.append("neutral")
            sentiment_score.append(0)
        else:
            model_output = sentiment_task(sentence)
            sentiment.append(model_output[0]["label"])
            sentiment_score.append(model_output[0]["score"])

    # Add labels and scores as new columns
    data["sentiment"] = sentiment
    data["sentiment_score"] = sentiment_score

    return data


ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
)

# Function to anonymize names in text


def anonymize_names_with_transformers(text):
    # Check if the text is empty or not a string
    if not text or not isinstance(text, str):
        return text  # Return the text as-is if it's invalid or empty

    # Initialize the anonymized text
    anonymized_text = text

    try:
        # Run the NER pipeline on the valid input text
        entities = ner_pipeline(text)

        # Iterate over detected entities
        for entity in entities:
            # Check if the entity is classified as a person
            if entity["entity_group"] == "PER":
                # Replace the detected name with a placeholder
                anonymized_text = anonymized_text.replace(entity["word"], "[PERSON]")
    except ValueError as e:
        # Log the error for debugging
        print(f"Error processing text: {text}")
        raise e

    return anonymized_text



# Zer0-shot classification - do_better column
def batch_generator(data, column_name, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[column_name][i : i + batch_size]
        # Logging the batch content; you can comment this out or remove it in production
        print(f"Batch starting at index {i}: {batch}")
        yield batch, i  # Yield the batch and the starting index


@time_it
def feedback_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification", model=model, tokenizer=tokenizer, device=-1
        )  # -1 for CPU
    except Exception as e:
        print(f"Error in initializing the classifier pipeline: {e}")
        return data  # Exit if the pipeline cannot be created

    # Define the categories for classification
    categories = [
        "Appointment Accessibility",
        "Reception Staff Interaction",
        "Medical Staff Competence",
        "Patient-Doctor Communication",
        "Follow-Up and Continuity of Care",
        "Facilities and Cleanliness",
        "Prescription and Medication Management",
        "Referral Efficiency",
        "Emergency Handling",
        "Patient Privacy and Confidentiality",
        "Telehealth Services",
        "Patient Education and Resources",
        "Waiting Room Comfort",
        "Patient Empowerment and Support",
        "Health Outcome Satisfaction",
        "Cultural Sensitivity",
        "Accessibility for Disabled Patients",
        "Mental Health Support",
        "Nursing Quality",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "Overall Patient Satisfaction",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
    ]  # Include all your categories here

    # Initialize the list to store labels
    feedback_labels = [""] * len(data)  # Pre-fill with empty strings

    # Process batches
    for batch, start_index in batch_generator(data, "free_text", batch_size):
        # Validate and filter batch data
        valid_sentences = [
            (sentence.strip(), idx)
            for idx, sentence in enumerate(batch)
            if isinstance(sentence, str) and sentence.strip()
        ]
        if not valid_sentences:
            continue  # Skip if no valid sentences are present

        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        try:
            # Perform classification
            model_outputs = classifier(list(sentences), categories, device="cpu")
            # Assign the most relevant category label
            for output, idx in zip(model_outputs, valid_indices):
                feedback_labels[start_index + idx] = output["labels"][0]
        except Exception as e:
            print(f"Error during classification: {e}")
            # Optionally, handle specific actions for failed classification, such as logging or retrying

    # Assign the computed labels back to the data
    data["feedback_labels"] = feedback_labels
    return data


@time_it
def improvement_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification", model=model, tokenizer=tokenizer, device=-1
        )  # -1 denotes CPU
    except Exception as e:
        print(f"Error initializing the classifier pipeline: {e}")
        return data  # Exit if the pipeline cannot be created

    # Define the labels for improvement categories
    improvement_labels_list = [
        "Appointment Accessibility",
        "Reception Staff Interaction",
        "Medical Staff Competence",
        "Patient-Doctor Communication",
        "Follow-Up and Continuity of Care",
        "Facilities and Cleanliness",
        "Prescription and Medication Management",
        "Referral Efficiency",
        "Emergency Handling",
        "Patient Privacy and Confidentiality",
        "Telehealth Services",
        "Patient Education and Resources",
        "Waiting Room Comfort",
        "Patient Empowerment and Support",
        "Health Outcome Satisfaction",
        "Cultural Sensitivity",
        "Mental Health Support",
        "Accessibility for Disabled Patients",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "Overall Patient Satisfaction",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
    ]  # Your improvement labels

    # Initialize the list to store improvement labels
    improvement_labels = [""] * len(data)  # Pre-fill with empty strings

    # Iterate over data in batches
    for batch, start_index in batch_generator(data, "do_better", batch_size):
        # Validate and filter batch data
        valid_sentences = [
            (sentence.strip(), idx)
            for idx, sentence in enumerate(batch)
            if isinstance(sentence, str) and sentence.strip()
        ]
        if not valid_sentences:
            continue  # Skip if no valid sentences are present

        sentences, valid_indices = (
            zip(*valid_sentences) if valid_sentences else ([], [])
        )

        try:
            # Classify the valid sentences
            model_outputs = classifier(
                list(sentences), improvement_labels_list, device="cpu"
            )
            # Update labels based on classification output
            for output, idx in zip(model_outputs, valid_indices):
                improvement_labels[start_index + idx] = output["labels"][0]
        except Exception as e:
            print(f"Error during classification: {e}")
            # Handle errors appropriately, possibly by logging or taking specific actions

    # Assign the computed labels back to the data
    data["improvement_labels"] = improvement_labels
    return data



@time_it
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


@time_it
def clean_data(df):
    # Copy the DataFrame to avoid modifying the original data
    cleaned_df = df.copy()
    # Apply the conditions and update the DataFrame
    cleaned_df.loc[cleaned_df["do_better_len"] < 6, "do_better"] = np.nan
    cleaned_df.loc[cleaned_df["free_text_len"] < 3, "free_text"] = np.nan
    
    return cleaned_df


@time_it
def concat_save_final_df(processed_df, new_df):
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    combined_data.sort_values(by="time", inplace=True, ascending=True)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", index=False)
    print(f"💾 data.csv saved to: {DATA_PATH}")


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


if __name__ == "__main__":

    monitor = cronitor.Monitor("UFDCXf")
    monitor.ping(state="run")
    logger.info("▶️ Friends & Family Test Analysis - MAKE DATA - Started")

    # Load new data from Google Sheet
    raw_data = load_google_sheet()
    logger.info("Google Sheet Data Loaded")

    # Load local data.csv to dataframe
    processed_data = load_local_data()
    logger.info("Data.csv Loadded")

    # Return new data for processing
    data = raw_data[~raw_data.index.isin(processed_data.index)]
    logger.info(f"🆕 New rows to process: {data.shape[0]}")

    if data.shape[0] != 0:
        data = word_count(data)  # word count
        data = add_rating_score(data)
        
        data = clean_data(data)
        
        logger.info("🫥 Annonymize free_text and do_better")
        data["free_text"] = data["free_text"].apply(anonymize_names_with_transformers)
        data["do_better"] = data["do_better"].apply(anonymize_names_with_transformers)
        data = feedback_classification(data, batch_size=16)
        
        data = sentiment_analysis(data)
        
        data = improvement_classification(data, batch_size=16)
        logger.info("Data pre-processing completed")
        
        logger.info("💾 Concat Dataframes to data.csv successfully")
        concat_save_final_df(processed_data, data)

        do_git_merge()  # Push everything to GitHub
        logger.info("Pushed to GitHub - Master Branch")
        monitor.ping(state="complete")
        logger.info("✅ Successful Run completed")
    else:
        monitor.ping(state="complete")
        print(f"{Fore.RED}[*] No New rows to add - terminated.")
        logger.error("❌ Make Data terminated - No now rows")
