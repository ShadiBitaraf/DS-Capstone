"""
#i may have to do words per review instead of "document"
feature extracting to get some meaningful insights of the data.

Extracted the following features:

Number of words in a document 
Number of characters in a document
Average word length of the document
Number of stop-words present
Number of numeric characters
Number of upper count characters
The polarity sentiment
"""

from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

STOP_WORDS = set(stopwords.words("english"))


def get_avg_word_len(x):
    """Get the average word length from a given sentence
    param x(str): the sentence of whose word length is to be taken
    return leng(numeric): the average word length"""

    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len / len(words)


def feature_extract(df, d):
    """Adds new columns in the given df, from the existing data
    count: number of words in the document (df[d])
    char count: number of characters in df[d]
    avg word_len: the average number of characters in the df[d]
    stop_words_len: number of stopwords present
    numeric_count: number of numeric characters present
    upper_counts: number of words in CAPS LOCK
    polarity: sentiment of the word, from -1(negative) to 1(positive)

    param df(dataframe): dataframe on which manipulation is to be done
    param d(str): column name in which the reuired words are present"""

    df["count"] = df[d].apply(lambda x: len(str(x).split()))  # correct
    df["char count"] = df[d].apply(lambda x: len(x.strip()))  # correct

    df["avg word_len"] = df[d].apply(lambda x: get_avg_word_len(x))
    df["stop_words_len"] = df[d].apply(
        lambda x: len([t for t in x.split() if t in STOP_WORDS])
    )
    df["numeric_count"] = df[d].apply(
        lambda x: len([t for t in x.split() if t.isdigit()])
    )
    df["upper_counts"] = df[d].apply(
        lambda x: len([t for t in x.split() if t.isupper() and len(x) > 3])
    )
    df["polarity"] = df["document"].map(lambda text: TextBlob(text).sentiment.polarity)


################# testing ####################

import pandas as pd

# Sample data
data = {
    "document": [
        "Another document for testing the function.",
        "       Sample document with repeated words.   ",
        "This is a sample document.",
        "This is a.",
    ]
}

# df = pd.DataFrame(data)

# # Call the feature_extract function
# feature_extract(df, "document")

# # Print the DataFrame to check the added columns
# print(df.head())


# # text = "This is a sample document.\n\n"
# # for char in text:
# #     print(f"{char}: {ord(char)}")
