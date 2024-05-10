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

import re
import unicodedata
import spacy

nlp = spacy.load("en_core_web_sm")


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


def make_to_base(x):
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
    return " ".join(x_list)


def preprocess(df, d):
    """Preprocesses the given document by applying the following functionalities
    lower: lowers all the characters for uniformity
    expansion: expands words like i'll to i will for better text classification
    remove special characters: using regex, removes all the punctuations etc
    remove space: removes trailing spaces and extra spaces between words
    remove accented characters: change accented characters to its normal equivalent
    remove stop words: removes the stop words in the sentence
    lemmatization: changes the words to their base form"""

    df[d] = df[d].apply(lambda x: x.lower())
    df[d] = df[d].apply(expand)
    df[d] = df[d].apply(lambda x: re.sub("[^A-Z a-z 0-9-]+", "", x))
    df[d] = df[d].apply(lambda x: " ".join(x.split()))
    df[d] = df[d].apply(lambda x: remove_accented_chars(x))
    df[d] = df[d].apply(lambda x: make_to_base(x))
    df[d] = df[d].apply(
        lambda x: " ".join([t for t in x.split() if t not in STOP_WORDS])
    )
