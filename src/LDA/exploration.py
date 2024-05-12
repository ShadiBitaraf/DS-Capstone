''''
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
'''



import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from plotly.offline import iplot
import plotly.graph_objs as go

# Assuming df is your DataFrame containing the data
df = pd.read_csv("your_dataset.csv")

# Sentiment Polarity Distribution
df['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

# Review Text Length Distribution
df['count'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Text Length Distribution')

# Word Cloud for Short Reviews
fig = plt.figure(figsize=(20, 8))
text = ' '.join(df.loc[df['count'] <= 10, 'document'].values)
wc = WordCloud(width=1000, 
               height=1000, 
               random_state=1, 
               background_color='Black',
               colormap='Set2',
               collocations=False).generate(text)
plt.imshow(wc)
plt.axis("off")
plt.show()

# Review Character Length Distribution
df['char count'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Character Length Distribution')

# Review Average Word Length Distribution
df['avg word_len'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Average Word Length Distribution')

# Review Numeric Count Distribution
df['numeric_count'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Numeric Count Distribution')

# Review Number of Uppercase Words Distribution
df['upper_counts'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Number of Uppercase Words Distribution')