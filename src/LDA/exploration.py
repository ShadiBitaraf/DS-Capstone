"""'
 This file will handle data exploration 
 and visualization tasks:

Load the IMDb dataset.
Perform data exploration.
Visualize sentiment polarity distribution.
Visualize review text length distribution.
Visualize review character length distribution.
Visualize review average word length distribution.
Visualize review numeric count distribution.
Visualize review number of uppercase words distribution.
"""

from dataset_loader import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from plotly.offline import iplot
#import plotly.io as pio
import plotly.graph_objs as go


def load_imdb_dataset(directory):
    """
    Load the IMDb dataset.
    """
    return load_dataset(directory)


def visualize_sentiment_polarity_distribution(df):
    """
    Visualize sentiment polarity distribution.
    """
    df["polarity"].iplot(
        kind="hist",
        bins=50,
        xTitle="polarity",
        linecolor="black",
        yTitle="count",
        title="Sentiment Polarity Distribution",
    )


def visualize_review_text_length_distribution(df):
    """
    Visualize review text length distribution.
    """
    df["count"].plot(
        kind="hist",
        bins=100,
        xTitle="review length",
        linecolor="black",
        yTitle="count",
        title="Review Text Length Distribution",
    )


def visualize_word_cloud_short_reviews(df):
    """
    Generate word cloud for short reviews.
    """
    text = " ".join(df.loc[df["count"] <= 10, "document"].values)
    wc = WordCloud(
        width=1000,
        height=1000,
        random_state=1,
        background_color="Black",
        colormap="Set2",
        collocations=False,
    ).generate(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def visualize_review_character_length_distribution(df):
    """
    Visualize review character length distribution.
    """
    df["char count"].iplot(
        kind="hist",
        bins=100,
        xTitle="review length",
        linecolor="black",
        yTitle="count",
        title="Review Character Length Distribution",
    )


def visualize_review_avg_word_length_distribution(df):
    """
    Visualize review average word length distribution.
    """
    df["avg word_len"].iplot(
        kind="hist",
        bins=100,
        xTitle="review length",
        linecolor="black",
        yTitle="count",
        title="Review Average Word Length Distribution",
    )


def visualize_review_numeric_count_distribution(df):
    """
    Visualize review numeric count distribution.
    """
    df["numeric_count"].iplot(
        kind="hist",
        bins=100,
        xTitle="review length",
        linecolor="black",
        yTitle="count",
        title="Review Numeric Count Distribution",
    )


def visualize_review_uppercase_word_distribution(df):
    """
    Visualize review number of uppercase words distribution.
    """
    df["upper_counts"].iplot(
        kind="hist",
        bins=100,
        xTitle="review length",
        linecolor="black",
        yTitle="count",
        title="Review Number of Uppercase Words Distribution",
    )


#################### testing #####################


def test_load_imdb_dataset():
    # Define a sample directory path
    directory = "/path/to/imdb/dataset"
    # Call the load_imdb_dataset function
    df = load_imdb_dataset(directory)
    # Check if DataFrame is loaded successfully
    assert isinstance(df, pd.DataFrame)
    # Check if DataFrame has expected columns
    assert all(col in df.columns for col in ["review", "sentiment"])
    print("Test for loading IMDb dataset passed successfully.")


def test_visualize_sentiment_polarity_distribution():
    # Define a sample DataFrame with necessary columns
    df = pd.DataFrame({"polarity": [1, 1, 0, -1, -1, 0]})
    # Call the visualization function
    visualize_sentiment_polarity_distribution(df)
    print("Test for visualizing sentiment polarity distribution passed successfully.")


def test_visualize_review_text_length_distribution():
    # Define a sample DataFrame with necessary columns
    df = pd.DataFrame({"count": [10, 20, 30, 40, 50]})
    # Call the visualization function
    visualize_review_text_length_distribution(df)
    print("Test for visualizing review text length distribution passed successfully.")


test_visualize_sentiment_polarity_distribution()
# Similarly, write test functions for other visualization functions
