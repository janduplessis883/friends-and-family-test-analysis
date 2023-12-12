import string
import re
import numpy as np
import time
from colorama import Fore, Back, Style, init
import functools

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

    remove_list = [
        "no",
        "none",
        "no...",
        "nothing",
        "nan",
        "an",
        "nope",
        "no everyting is fine",
        "keep it up",
        "no answer",
        "wait time",
        "yes",
        "no fine",
        "not really",
        "no nothinh",
        "not really",
        "nothings",
        "all good",
        "reminder",
        "na",
        "no really",
        "all good",
        "no ",
        "not at all",
        "excellent",
        "thanks",
        "not really",
        "thank you",
        "thank you",
        "it was good",
        "thats all",
        "no nothing",
        "no tks",
        "nathing",
        "not really",
        "yes",
        "its  okey",
        "not really",
        "answer above",
    ]

    # Replace 'no' with None
    if cleaned_text in remove_list:
        return ""
    else:
        return cleaned_text


# = Decorators =================================================================


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"{Fore.RED}{Back.LIGHTYELLOW_EX}[F] FUCTION: {func.__name__}()")
        result = func(*args, **kwargs)
        print(
            f"{Fore.GREEN}{Style.DIM}✅ Completed: {func.__name__}() - Time taken: {time.time() - start_time:.2f} seconds"
        )
        return result

    return wrapper


def debug_info(func):
    """Decorator that prints function execution information."""

    @functools.wraps(func)
    def wrapper_debug_info(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")

        # Get the start time
        start_time = time.perf_counter()

        value = func(*args, **kwargs)

        # Get the end time
        end_time = time.perf_counter()

        # Calculate the execution time
        run_time = end_time - start_time

        print(
            f"Finished {func.__name__!r} in {run_time:.4f} secs with result {value!r}"
        )
        return value

    return wrapper_debug_info
