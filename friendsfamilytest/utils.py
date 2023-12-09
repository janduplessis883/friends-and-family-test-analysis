import string
import re
import numpy as np


# Function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def clean_and_replace(text):
    text = remove_emojis(str(text))
    # Convert to lowercase and strip whitespace
    cleaned_text = str(text).lower().strip()

    # Remove punctuation and digits
    cleaned_text = cleaned_text.translate(
        str.maketrans("", "", string.punctuation + string.digits)
    )

    remove_list = ["no", "none", "no...", "nothing", "nan", "an", "nope"]
    # Replace 'no' with None
    if cleaned_text in remove_list:
        return ""
    else:
        return cleaned_text
    
# = Decorators =================================================================

import time
from colorama import Fore, Back, Style, init
init(autoreset=True)

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"{Fore.WHITE}{Back.BLACK}[FUNCTION] {func.__name__}() <Timer>")
        result = func(*args, **kwargs)
        print(f"{Fore.BLACK}{Back.GREEN}[✅] Completed - Time taken: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

# Now, decorate your functions with @time_it
@time_it
def text_classification(data):
    # Your existing code for text classification
    pass

@time_it
def sentiment_analysis(data):
    # Your existing code for sentiment analysis
    pass
