"""
Since LDA has an inbuilt TF-IDF vectorizer, we will have to use Count vectorizer.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_bow(df, d, range_min, range_max, mindf, maxdf):
    """Returns the count vectorized dataframe based on the arguments

    param df(dataframe): dataframe containing the values
    param  d(str): the column name, under which the documents are present
    param range_min(int): smallest n of n-gram wanted
    param range_max(int): largest n of n-gram wanted
    param mindf(int): threshold for common words
    param maxdf(int): threshold for rare words

    return cv(CountVectorizer): the count vectorizer
           cv_df(dataframe): dataframe containing the bag of words"""

    cv = CountVectorizer(ngram_range=(range_min, range_max), min_df=mindf, max_df=maxdf)
    text_counts = cv.fit_transform(df[d])
    cv_df = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names())
    return cv, cv_df
