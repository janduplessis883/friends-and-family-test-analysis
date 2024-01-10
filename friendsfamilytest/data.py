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


from friendsfamilytest.params import *
from friendsfamilytest.utils import *
from friendsfamilytest.auto_git.git_merge import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
init(autoreset=True)
warnings.filterwarnings("ignore")
secret_path = os.getenv("SECRET_PATH")
from sheethelper import *


@time_it
def load_google_sheet():
    sh = SheetHelper(
        sheet_url="https://docs.google.com/spreadsheets/d/1K2d32XmZQMdGLslNzv2ZZoUquARl6yiKRT5SjUkTtIY/edit#gid=1323317089",
        sheet_id=0,
    )
    data = sh.gsheet_to_df()
    data.columns = ["time", "rating", "free_text", "do_better"]
    data["time"] = pd.to_datetime(data["time"], format="%d/%m/%Y %H:%M:%S")

    return data


@time_it
def word_count(df):
    df["free_text_len"] = df["free_text"].str.split().apply(len)
    df["do_better_len"] = df["do_better"].str.split().apply(len)
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


@time_it
def anonymize(df):
    # List of surnames to look for
    surnames_to_find = [
        "burhan",
        "adib",
        "emiliani",
        "alex",
        "florko",
        "florka",
        "lula",
        "joyce",
        "christine",
        "jan",
        "orietta",
    ]

    # Function to replace surnames in text
    def replace_surname(text):
        for surname in surnames_to_find:
            # Create a regular expression pattern for the surname
            pattern = r"\b" + re.escape(surname) + r"\b"
            # Replace the surname with its first letter and a period
            text = re.sub(pattern, surname[0], text)
        return text

    # Apply the function to the 'free_text' column
    df["free_text"] = df["free_text"].apply(replace_surname)
    df["do_better"] = df["do_better"].apply(replace_surname)
    return df


@time_it
def textblob_sentiment(data):
    data["free_text"] = data["free_text"].fillna("").astype(str)

    def analyze_sentiment(text):
        if text:
            sentiment = TextBlob(text).sentiment
            return pd.Series(
                [sentiment.polarity, sentiment.subjectivity],
                index=["polarity", "subjectivity"],
            )
        else:
            return pd.Series([0, 0], index=["polarity", "subjectivity"])

    sentiments = data["free_text"].apply(analyze_sentiment)
    data = pd.concat([data, sentiments], axis=1)

    # Check if the number of rows matches
    if len(sentiments) != len(data):
        raise ValueError("Mismatched row count between original data and sentiments")

    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Define the function to be applied to each row
    def get_sentiment(row):
        # Analyze sentiment using SentimentIntensityAnalyzer
        score = sia.polarity_scores(row["free_text"])

        # Assign the scores to the row
        for key in ["neg", "neu", "pos", "compound"]:
            row[key] = score[key]

        # Determine the overall sentiment based on the scores
        row["sentiment"] = "neutral"  # Default to neutral
        if score["neg"] > score["pos"]:
            row["sentiment"] = "negative"
        elif score["pos"] > score["neg"]:
            row["sentiment"] = "positive"

        return row

    # Apply the function to each row
    data = data.apply(get_sentiment, axis=1)

    return data


# Zer0-shot classification - do_better column
def batch_generator(data, column_name, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[column_name][
            i : i + batch_size
        ], i  # Yield the batch and the starting index


# Zero-Shot Classification (facebook model tried), now review the BartForConditionalGeneration
# facebook/bart-large-mnli
# trl-internal-testing/tiny-random-BartForConditionalGeneration ❌
# ybelkada/tiny-random-T5ForConditionalGeneration-calibrated
# MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli


@time_it
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
        "Reception Staff",
        "Ambiance of Facility",
        "Facility Modernization and Upgrades",
        "Nursing Quality",
        "Waiting Times",
        "Referral Process",
        "Staffing Levels",
        "Facility Accessibility",
        "Poor Communication",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "After-Hours Service",
        "Staff Training and Development",
        "Prescription Process",
        "Quality of Medical Advice",
        "Overall Patient Satisfaction",
        "Appointment System Efficiency",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Mental Health Services",
        "Social Prescribing Services",
        "Chronic Disease Management",
        "No Improvement Suggestion",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
        "Clinical Pharmacist",
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


def openai_classify_string(input_string):
    prompt = """you are an expert practice manager for a GP Surgery, you will review improvement suggestions from patients and classify them into one of the following categories:
improvement_labels_list = [
        "Reception Staff",
        "Ambiance of Facility",
        "Facility Modernisation and Upgrades",
        "Nursing Quality",
        "Waiting Times",
        "Referral Process",
        "Staffing Levels",
        "Facility Accessibility",
        "Poor Communication",
        "Online Services & Digital Health",
        "Patient Safety",
        "Weekend Service Availability",
        "Telephone Service",
        "After-Hours Service",
        "Staff Training and Development",
        "Prescription Process",
        "Quality of Medical Advice",
        "Overall Patient Satisfaction",
        "Appointment System Efficiency",
        "Blood Test Results & Imaging",
        "Patient Participation Group",
        "Mental Health Services",
        "Social Prescribing Services",
        "Chronic Disease Management",
        "No Improvement Suggestion",
        "Doctor Consultations",
        "Home Visits",
        "Cancer Screening",
        "Vaccinations",
        "Test Results",
        "Clinical Pharmacist",
    ]
when you respond only select the most appropriate category and only return the category as specified in the 'improvement_labels_list', do not return any other text. if the text provided does not fit into an improvement suggestion category classify it as 'No Improvement Suggestion'
You should only ever return one of the 'improvement_labels_list' classifications or nothing at all. This is very important. Positive comments without any improvement suggestions should be classified as 'Overall Patient Satisfaction', and phrases that are very short and has no meaning should be classified as 'No Improvement Suggestion'. """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_string},
        ],
    )

    gpt3_classification = completion.choices[0].message.content
    print(
        f"[GPT3 working] - {Fore.LIGHTGREEN_EX}{input_string} ::: {Fore.BLUE}{gpt3_classification}"
    )

    return gpt3_classification


@time_it
def gpt3_improvement_classification(df):
    do_better_list = df["do_better"].tolist()
    gpt3_labels = []

    for input in do_better_list:
        if pd.isna(input):
            gpt3_labels.append("")  # Append an empty string for NaN values
        else:
            # gpt3_classification = openai_classify_string(input)
            gpt3_labels.append("gpt3_classification")  # Append classification label

    df["improvement_gpt3"] = gpt3_labels

    return df


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
def concat_save_final_df(processed_df, new_df):
    combined_data = pd.concat([processed_df, new_df], ignore_index=True)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", index=False)
    print(f"💾 data.csv saved to: {DATA_PATH}")


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


if __name__ == "__main__":
    print(f"{Fore.WHITE}{Back.BLACK}[+] Friends & Family Test Analysis - MAKE DATA")

    # Load new data from Google Sheet
    raw_data = load_google_sheet()

    # Load local data.csv to dataframe
    processed_data = load_local_data()

    # Return new data for processing
    data = raw_data[~raw_data.index.isin(processed_data.index)]

    print(f"{Fore.BLUE}[*] New rows to process: {data.shape[0]}")
    if data.shape[0] != 0:
        data = clean_text(data)  # clean text
        data = word_count(data)  # word count
        data = add_rating_score(data)
        data = anonymize(data)
        data = text_classification(data)
        data = sentiment_analysis(data)
        data = improvement_classification(
            data, batch_size=16
        )  # data = gpt3_improvement_classification(data)
        data = textblob_sentiment(data)
        concat_save_final_df(processed_data, data)
        do_git_merge()  # Push everything to GitHub
    else:
        print(f"{Fore.RED}[*] No New rows to add - terminated.")
