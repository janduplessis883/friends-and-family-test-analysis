import string
import re
import numpy as np

# Function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_and_replace(text):
    text = remove_emojis(str(text))
    # Convert to lowercase and strip whitespace
    cleaned_text = str(text).lower().strip()

    # Remove punctuation and digits
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation + string.digits))

    remove_list = ['no', 'none', 'no...', 'nothing', 'nan', 'an', 'nope']
    # Replace 'no' with None
    if cleaned_text in remove_list:
        return ''
    else:
        return cleaned_text