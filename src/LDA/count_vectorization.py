"""
Since LDA has an inbuilt TF-IDF vectorizer, we will have to use Count vectorizer.
It transforms the text data into a count vectorized format, where each row 
represents a document and each column represents a unique word or n-gram in the documents.
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
               cv_df(dataframe): dataframe containing the bag of words

    ----------------
      Inputs:
    - df (DataFrame): A pandas DataFrame containing the text data.
    - d (str): The column name in the DataFrame df where the text documents are stored.
    - range_min (int): The smallest n-gram (sequence of n words) wanted.
    - range_max (int): The largest n-gram wanted.
    - mindf (int): The threshold for common words. It discards terms that appear in less than this proportion of documents.
    - maxdf (int): The threshold for rare words. It discards terms that appear in more than this proportion of documents.

    Outputs:
    - cv (CountVectorizer): The configured CountVectorizer object.
           This object contains the configuration used for vectorization.
    - cv_df (DataFrame): A pandas DataFrame containing the count vectorized
           representation of the text data. Each row represents a document,
           and each column represents a unique word or n-gram in the documents.
           The values in the DataFrame represent the count of each word or n-gram in the respective document.
    """

    cv = CountVectorizer(ngram_range=(range_min, range_max), min_df=mindf, max_df=maxdf)
    text_counts = cv.fit_transform(df[d])
    cv_df = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names_out())
    return cv, cv_df


"""
The specification for vectorizing are:

minimum n gram - 3
maximum n gram - 4
min_df - 50%
max_df - 5%
"""


################### testing ####################

# # Create a sample dataframe
# data = {
#     "text": [
#         "This is a sample document.",
#         "Another document for testing the function.",
#         "Sample document with repeated words.",
#     ]
# }
# df = pd.DataFrame(data)

# # Call the function with sample data
# cv, cv_df = get_bow(df, "text", 3, 4, 0.05, 0.5)

# # Check if the CountVectorizer object is created
# print("CountVectorizer Object:", cv)

# # Check if the dataframe containing bag of words is created
# print("Bag of Words DataFrame:")
# print(cv_df.head())
