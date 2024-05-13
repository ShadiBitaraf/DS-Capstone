"""
we are doing the following:

Made all the characters to lower case
Expanded the short forms, like I’ll → I will
Removed special characters
Removed extra and trailing spaces
Removed accented characters and replaced them with their alternative
Lemmatized the words
Removed stop words
"""

from nltk.corpus import stopwords
import re
import unicodedata
import spacy


STOP_WORDS = set(stopwords.words("english"))  # Load English stopwords


nlp = spacy.load("en_core_web_sm")


# list of contractions and their related expansions (from web)
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "'ll": " will",
    "'ve": " have",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and ",
    "tbh": "to be honest",
}


def expand(x):
    """Some of the words like 'i'll', are expanded to 'i will' for better text processing
    The list of contractions is taken from the internet

    param x(str): the sentence in which contractions are to be found and expansions are to be done

    return x(str): the expanded sentence"""
    if type(x) == str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x


def remove_accented_chars(x):
    """The function changes the accented characters into their equivalent normal form,
    to do so, normalize function with 'NFKD' is used which replaces the compatibility characters into
    theri euivalent

    param x(str): the sentence in which accented characters are to be detected and removes
    return x(str): sentence with accented characters replaced by their equivalent"""

    x = (
        unicodedata.normalize("NFKD", x)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return x


def make_to_base(x):  # TODO little spacing issue: would 've instead of would've
    """Converting the words to their base word and dictionary head word i.e to lemmatize
    param x(str): the sentence in which the words are to be converted (lemmatization)
    return x(str): the lemmatized sentence"""

    x_list = []
    doc = nlp(x)

    for token in doc:
        lemma = str(token.lemma_)

        # in spacy, words like I, you are lemmatized as -PRON- and are,and etc are lemmatized to be,
        # since these words are present widely, we keep them as the original.
        # Anyways the words will be removed as stop words later

        if lemma == "-PRON-" or lemma == "be":
            lemma = token.text
        x_list.append(lemma)

        # Join the list elements with a space between each element, except for the last element and punctuation at the end
    lemmatized_sentence = ""
    for i, word in enumerate(x_list):
        if i == len(x_list) - 1:  # Last element
            lemmatized_sentence += word
        elif (
            i + 1 < len(x_list) and x_list[i + 1] in ",.!?;:"
        ):  # Check if the next element is a punctuation mark
            lemmatized_sentence += word
        else:
            lemmatized_sentence += word + " "

    return lemmatized_sentence


def preprocess(df, d):
    """Preprocesses the given document by applying the following functionalities
    lower: lowers all the characters for uniformity
    replace: removes any breaks in the review
    expansion: expands words like i'll to i will for better text classification
    remove special characters: using regex, removes all the punctuations etc
    remove space: removes trailing spaces and extra spaces between words
    remove accented characters: change accented characters to its normal equivalent
    remove stop words: removes the stop words in the sentence
    lemmatization: changes the words to their base form
    remove movie/film"""

    df[d] = df[d].apply(lambda x: x.lower())
    df[d] = df[d].str.replace('<br /><br />', '')
    df[d] = df[d].apply(expand)
    df[d] = df[d].apply(lambda x: remove_accented_chars(x))
    df[d] = df[d].apply(lambda x: re.sub("[^A-Z a-z 0-9-]+", "", x))
    df[d] = df[d].apply(lambda x: " ".join(x.split()))
    df[d] = df[d].apply(lambda x: make_to_base(x))
    df[d] = df[d].apply(
        lambda x: " ".join([t for t in x.split() if t not in STOP_WORDS]))
    df[d] = df[d].str.replace('movie', '')
    df[d] = df[d].str.replace('film', '')


##################### TESTING #######################


# Sample input data
# input_sentence = "I'll GO TO café tomorrow. i would've? cliché she's: o'clock!"
# print("input:\n" + input_sentence)

# # 1. Test the expand function
# expanded_sentence = expand(input_sentence)
# print("\nExpanded sentence:\n", expanded_sentence)

# # 2. Test the remove_accented_chars function
# accented_removed_sentence = remove_accented_chars(input_sentence)
# print("\nAccented characters removed:\n", accented_removed_sentence)

# 3. Test the make_to_base function
# lemmatized_sentence = make_to_base(input_sentence)
# print("\nLemmatized sentence:\n", lemmatized_sentence)

# # You can test the preprocess function if you have a DataFrame and column name to preprocess
# # For example:
# import pandas as pd

# df = pd.DataFrame({"text_column": [input_sentence]})

# preprocess(df, "text_column")
# print("\nPreprocessed DataFrame aka function output:", df)
