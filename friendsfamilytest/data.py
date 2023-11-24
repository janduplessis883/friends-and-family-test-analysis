import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from transformers import pipeline
from sheethelper import SheetHelper
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
from friendsfamilytest.params import *
import warnings
import subprocess

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
    data["full_text"] = (
        data["free_text"].astype("str")
        + " Anything we can do to make your experience better? "
        + data["do_better"].astype("str")
    )
    data["full_text"] = (
        data["full_text"].str.replace("\s+", " ", regex=True).str.strip()
    )
    return data


def text_classification(data):
    # Initialize classifier
    classifier = pipeline(
        task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None
    )

    # Initialize lists to store labels and scores
    labels1 = []
    scores1 = []
    labels2 = []
    scores2 = []

    # Iterate over DataFrame rows and classify text
    for _, row in data.iterrows():
        sentence = row["full_text"]
        model_outputs = classifier(sentence)
        labels1.append(model_outputs[0][0]["label"])
        scores1.append(model_outputs[0][0]["score"])
        labels2.append(model_outputs[0][1]["label"])
        scores2.append(model_outputs[0][1]["score"])

    # Add labels and scores as new columns
    data["label1"] = labels1
    data["score1"] = scores1
    data["label2"] = labels2
    data["score2"] = scores2

    return data


def sentinment_analysis(data):
    sentiment_task = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # Initialize lists to store labels and scores
    labels3 = []
    scores3 = []

    # Iterate over DataFrame rows and classify text
    for _, row in data.iterrows():
        sentence = row["full_text"]
        model_outputs = sentiment_task(sentence)
        labels3.append(model_outputs[0]["label"])
        scores3.append(model_outputs[0]["score"])

    # Add labels and scores as new columns
    data["label3"] = labels3
    data["score3"] = scores3

    return data


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
    print(f"{Fore.WHITE}{Back.BLACK}[>] Creating New data.csv from Google Sheet")
    
    data = load_google_sheet()
    print(f"{Fore.RED}[+] Google Sheet Loaded")
    
    data = text_classification(data)
    print(f"{Fore.BLUE}[+] Text Classification completed")
    
    data = sentinment_analysis(data)
    print(f"{Fore.BLUE}[+] Sentiment Analysis completed")
    
    data = add_rating_score(data)
    print(f"{Fore.BLUE}[+] Rating score added")
    
    print(f"{Fore.GREEN}[+] Data Successfully Loaded")
    
    data.to_csv(f"{DATA_PATH}/data.csv", index=False)
    print(f"{Fore.YELLOW}[i] 💾 Data saved to '/data/data.csv'")

    print(f"{Fore.WHITE}{Back.BLACK}[>] Git: Push to GitHub Repo")

    # Set the path to your local git repository
    repo_path = LOCAL_GIT_REPO
    # Set the remote name and branch name
    remote = "origin"
    branch = "master"
    # Change to the repo directory
    os.chdir(repo_path)

    # Get the latest commits
    subprocess.run(["git", "fetch", remote])
    print(f"{Fore.RED}[+] Git: fetch remote")
    
    # Checkout the desired branch
    #subprocess.run(["git", "checkout", branch])

    # Add all changes
    subprocess.run(["git", "add", "."])

    # Commit changes with a message
    message = "Automated commit from Python script"
    subprocess.run(["git", "commit", "-m", message])
    print(f"{Fore.RED}[+] Git: commit")
    
    # Push changes
    subprocess.run(["git", "push", remote, branch])
    print(f"{Fore.RED}[+] Git: push to remote {branch}")
    print(f"{Fore.YELLOW}[i] ✅ data.csv push to GitHub successful")
