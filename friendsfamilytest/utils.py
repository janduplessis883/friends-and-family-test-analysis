import string
import re
import numpy as np
import time
from colorama import Fore, Back, Style, init

init(autoreset=True)


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

    remove_list = ["no", "none", "no...", "nothing", "nan", "an", "nope", "no everyting is fine"]
    # Replace 'no' with None
    if cleaned_text in remove_list:
        return ""
    else:
        return cleaned_text


# = Decorators =================================================================

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"🕥{Fore.RED}{Style.BRIGHT}---FUCTION: {func.__name__}()")
        result = func(*args, **kwargs)
        print(
            f"{Fore.GREEN}{Style.DIM}✅-Completed: {func.__name__}() - Time taken: {time.time() - start_time:.2f} seconds"
        )
        return result

    return wrapper
