'''
This file will handle visualizing the LDA topics and other relevant visualizations:

Visualize the distribution of assigned topics.
Generate word clouds for each topic.
'''


# Use WordCloud library to create visualizations

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordcloud(topic_words):
    fig = plt.figure(figsize=(80, 32))
    for i, top_words_topic in enumerate(topic_words):  # Iterate through each list of top words for a topic
        ax = fig.add_subplot(4, 3, i + 1)
        text = ' '.join(top_words_topic)  # Join the top words into a single string
        wc = WordCloud(width=1000,
                    height=1000,
                    random_state=1,
                    background_color='gray',  # Changed background color for visibility
                    colormap='Set2',  # You can adjust the colormap if needed
                    collocations=False).generate(text)
        ax.imshow(wc)
        ax.set_title(f'Topic {i}')
        ax.axis("off")
    
    plt.show()
