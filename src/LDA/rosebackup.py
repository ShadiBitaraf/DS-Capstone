'''
This file will handle the topic assignment using LDA.

Load the IMDb dataset.
Preprocess the text data.
Use LDA to assign topics to the reviews.
Print the top 10 words per topic.
Assign topics to the reviews based on LDA results.
Store the assigned topics in the DataFrame.

'''

'''
Steps to assign topics to each document:
1)Randomly initialize each word to a topic amongst the K topics (# of pre-defined topics)

2)For each document d, compute:
Proportion of words in document d that are assigned to topic t
Proportion of assignments to topic t across all documents from words that come from w

3)Reassign topic T' to word w with probability p(t'|d)*p(w|t') considering all other words and their topic assignments

Repeat 3 until we reach a steady state

'''

# Will organize imports later:
import pandas as pd
import preprocessing
import count_vectorization
from dataset_loader import load_dataset
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
STOP_WORDS = list(set(stopwords.words("english")))

#ADDING PRINT CHECK STATEMENTS --> file takes a long time to load, so using the checks to see where it's pausing
# Load IMDb dataset
traindata = load_dataset("/Users/roseabsin/Desktop/DS-Capstone/data/raw/Imdb/train")
print("imdb dataset loaded")
traindata = traindata.iloc[0:1000].copy() #DELETE THIS LINE FOR FULL FILE
print(traindata.head(10))
# Preprocess text data 

preprocessing.preprocess(traindata, "review")
print("preprocessing done:")
print(traindata.head(10))

cv, cv_df = count_vectorization.get_bow(traindata, "review", 3, 4, 0.01, 0.9)
print("count vectorization retreived")
vect =TfidfVectorizer(stop_words=STOP_WORDS,max_features=1000)
print("tfid vectorizer made")
vect_text=vect.fit_transform(traindata['review'])
print("preprocessing done")

# Assign topics to the reviews
lda_model=LatentDirichletAllocation(n_components=10,
learning_method='online',random_state=42,max_iter=1) 
lda_top=lda_model.fit_transform(vect_text)
print("lda model made")

# Testing
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")

vocab = vect.get_feature_names_out()

topicwords = []
for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]

    topicwords.append([word for word, _ in sorted_words])
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


print("End of file, done running")

import visualization
visualization.create_wordcloud(topicwords)
