import string
import re
import numpy as np
import time
from colorama import Fore, Back, Style, init
import functools
import streamlit as st
from loguru import logger

init(autoreset=True)





def clean_and_replace(text):
    # Convert to lowercase and strip whitespace
    cleaned_text = str(text).strip()

    # Remove punctuation and digits
    # cleaned_text = cleaned_text.translate(
    #     str.maketrans("", "", string.punctuation + string.digits)
    # )

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
        "no comment",
    ]

    # Replace 'no' with None
    if cleaned_text in remove_list:
        return ""
    else:
        return cleaned_text


def sentiment_totals(data):
    total_pos = len(data.loc[data["sentiment"] == "positive"])
    total_neg = len(data.loc[data["sentiment"] == "negative"])
    total_neu = len(data.loc[data["sentiment"] == "neutral"])
    total_data = len(data)
    return [
        round(total_pos / total_data, 2),
        round(total_neu / total_data, 2),
        round(total_neg / total_data, 2),
    ]





# = Decorators =================================================================


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        func_name = func.__name__
        logger.info(f"Function '{func_name}' ⚡️{elapsed_time:.6f} sec")
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


# end of file
