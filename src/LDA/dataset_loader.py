import os
import pandas as pd


def load_dataset(directory):
    data = []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), "r", encoding="utf-8") as file:
                text = file.read()
                data.append((text, label))
    return pd.DataFrame(data, columns=["review", "sentiment"])


"""
Based on the README of the dataset, we have two directories: train and test, 
each containing pos and neg subdirectories representing positive and negative reviews, 
respectively. Therefore, to test the load_dataset function, 
use any file from these directories.
"""


# def test_load_dataset():
#     directory = "/Users/shadibitarafhaghighi/Desktop/DS-Capstone/data/raw/Imdb/train"  # Replace this with the actual path to your dataset directory
#     df = load_dataset(directory)
#     print("Dataset loaded successfully!")
#     print("Shape of the DataFrame:", df.shape)
#     print("Sample of the DataFrame:")
#     print(df.head())


# test_load_dataset()
