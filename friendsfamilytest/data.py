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

    return data


@time_it
def word_count(df):
    df["free_text_len"] = df["free_text"].apply(lambda x: len(str(x).split()) if isinstance(x, str) else np.nan)

    df["do_better_len"] = df["do_better"].apply(lambda x: len(str(x).split()) if isinstance(x, str) else np.nan)

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
            dataframe.at[index, column_name] = ''

    return dataframe

@time_it
def sentiment_analysis(data):
    sentiment_task = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # Initialize lists to store labels and scores
    sentiment = []
    sentiment_score = []

    # Iterate over DataFrame rows and classify text
    for index, row in data.iterrows():
        print(index)
        sentence = row["free_text"]
        sentence = str(sentence)
        sentence = sentence[:513]
        if sentence == 'nan':
            sentiment.append('neutral')
            sentiment_score.append(1)
        else:
            model_output = sentiment_task(sentence)
            sentiment.append(model_output[0]['label'])
            sentiment_score.append(model_output[0]['score'])
        
    # Add labels and scores as new columns
    data["sentiment"] = sentiment
    data["sentiment_score"] = sentiment_score

    return data

ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Function to anonymize names in text
@time_it
def anonymize_names_with_transformers(text):
    # Check if the text is empty and return an empty string if so
    if pd.isnull(text) or text.strip() == "":
        return ""

    # Run the NER pipeline on the input text
    entities = ner_pipeline(text)
    anonymized_text = text

    # Iterate over detected entities
    for entity in entities:
        # Check if the entity is a person
        if entity['entity_group'] == 'PER':
            # Replace the detected name with [PERSON]
            anonymized_text = anonymized_text.replace(entity['word'], '[PERSON]')

    return anonymized_text




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
        "Ambiance of Facility",
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


@time_it
def feedback_classification(data, batch_size=16):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    # Create classifier pipeline
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    # Labels
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
        "Ambiance of Facility",
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
    ]

    # Initialize the list to store labels
    feedback_labels = [""] * len(data)  # Pre-fill with empty strings

    # Iterate over batches
    for batch, start_index in batch_generator(data, "free_text", batch_size):
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
            model_outputs = classifier(list(sentences), categories, device="cpu")
            # Assign labels to corresponding indices
            for output, idx in zip(model_outputs, valid_indices):
                feedback_labels[start_index + idx] = output["labels"][0]
                print(
                    f"{Fore.GREEN}Batch processed: {start_index + idx + 1}/{len(data)}"
                )

    # Add labels as a new column
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
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    # Labels
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
        "Ambiance of Facility",
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
    combined_data.sort_values(by='time', inplace=True, ascending=True)
    combined_data.to_csv(f"{DATA_PATH}/data.csv", index=False)
    print(f"💾 data.csv saved to: {DATA_PATH}")


@time_it
def load_local_data():
    df = pd.read_csv(f"{DATA_PATH}/data.csv")
    df["time"] = pd.to_datetime(df["time"], dayfirst=True)
    return df


if __name__ == "__main__":

    monitor = cronitor.Monitor('UFDCXf')
    monitor.ping(state='run')
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
        data = clean_text(data)  # clean text
        data = word_count(data)  # word count
        data = add_rating_score(data)
        logger.info(f"4️⃣ Discard short input")
        data = check_column_length(data, 'do_better', 5)
        data = check_column_length(data, 'free_text', 2)
        data["free_text"] = data["free_text"].apply(anonymize_names_with_transformers)
        data["do_better"] = data["do_better"].apply(anonymize_names_with_transformers)
        data = feedback_classification(data, batch_size=16)
        data = sentiment_analysis(data)
        data = improvement_classification(
            data, batch_size=16
        )  # data = gpt3_improvement_classification(data)
        data = textblob_sentiment(data)
        logger.info("Data pre-processing completed")
        
        concat_save_final_df(processed_data, data)
        logger.info("💾 Concat Dataframes to data.csv successfully")
        
        do_git_merge()  # Push everything to GitHub
        logger.info("Pushed to GitHub - Master Branch")
        monitor.ping(state='complete')
        logger.info("✅ Successful Run completed")
    else:
        monitor.ping(state='complete')
        print(f"{Fore.RED}[*] No New rows to add - terminated.")
        logger.error("❌ Make Data terminated - No now rows")
        
