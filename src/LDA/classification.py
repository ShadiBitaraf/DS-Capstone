"""
This file will handle the classification task using a Naive Bayes classifier.

Load the IMDb dataset.
Encode the topics for classification.
Split the dataset into training and testing sets.
Define a pipeline for text classification.
Test and score the classification model for each topic.

"""
#Load the IMDb dataset
from dataset_loader import load_dataset
import os

traindata = load_dataset(os.path.join(os.getcwd(), "data/raw/Imdb/train"))

# Encode topics for classification

