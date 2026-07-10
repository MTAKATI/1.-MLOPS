# Topic Modeling
Large amounts of data are collected everyday. As more information becomes available, it becomes difficult to access what we are looking for. So, we need tools and techniques to organize, search and understand vast quantities of information.
Topic modelling provides us with methods to organize, understand and summarize large collections of textual information. It helps in:

•	Discovering hidden topical patterns that are present across the collection

•	Annotating documents according to these topics

•	Using these annotations to organize, search and summarize texts


Topic modelling can be described as a method for finding a group of words (i.e topic) from a collection of documents that best represents the information in the collection. It can also be thought of as a form of text mining – a way to obtain recurring patterns of words in textual material.
In this project we will use unsupervised techniques, to cluster/ group reviews to identify main topics/ ideas in the sea of text. This will be applicable to any textual reviews but we can focus on twitter data which is more real world and more complex than reviews obtained from review or survey forms. 



import nltk
import numpy as np
import re # remove regex
import pandas as pd 

%matplotlib inline


## Get multiple outputs in the same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

## Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
## Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
raw_data  = pd.read_csv("tweets.csv",encoding = 'ISO-8859-1')
print(len(raw_data))
df = raw_data
df

unique_text = df.tweet.unique()
print(len(unique_text))
df.head()

df['tweet'][444]

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

### Remove any @ mentions
df['Clean_text'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
df['Clean_text']
df['Clean_text'] = df['Clean_text'].str.replace("[^a-zA-Z#]", " ")
df['Clean_text']
df["Clean_text"]= df["Clean_text"].str.lower() 
df['Clean_text']
df['Clean_text'] = df['Clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
df.head()
tokenized_tweet = df['Clean_text'].apply(lambda x: x.split())
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['Clean_text'] = tokenized_tweet
df
df.loc[:,('Clean_text')]
df.drop_duplicates(subset=['Clean_text'], keep = 'first',inplace= True)

df.reset_index(drop=True,inplace=True)
df
df['Clean_text_length'] = df['Clean_text'].apply(len)
df.head()
df[df['Clean_text_length']==0]
raw_data[raw_data['username']=='omanmessi']
df[df['Clean_text_length']==0]['Clean_text'] ## Looks like these are tweets with different languages or just hastags.
# We can simply drop these tweets
indexes = df[df['Clean_text_length']==0]['Clean_text'].index
indexes

df.drop(index = indexes,inplace=True)
df.info()
df.reset_index(drop=True,inplace=True)
df.info()
df['Clean_text'].head()
## Vectorizer
[Hi how are you]
[Hi how do you do]

[Hi, how, are, you, do]
[1,1,1,1,0]
[1,1,0,1,2]
TfidfVectorizer:

Convert a collection of raw documents to a matrix of TF-IDF features.
In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

CountVectorizer:

Convert a collection of text documents to a matrix of token counts
The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary\
Number of features will be equal to the vocabulary size found by analyzing the data.
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word',ngram_range=(1,1), stop_words='english', min_df = 0.0001, max_df=0.7)
count_vect.fit(df['Clean_text'])
desc_matrix = count_vect.transform(df["Clean_text"])
desc_matrix
desc_matrix.toarray()
desc_matrix.shape
## Clustering
!pip3 install KMeans
!pip3 install wordcloud

from sklearn.cluster import KMeans
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests
import matplotlib.pyplot as plt

num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()

tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
frame = pd.DataFrame(tweets, index = [clusters])
frame
frame.head()
frame['Cluster'].value_counts()
cluster_0=frame[frame['Cluster'] == 0]
cluster_0
def wordcloud(cluster):
  # combining the image with the dataset
  Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

  # We use the ImageColorGenerator library from Wordcloud 
  # Here we take the color of the image and impose it over our wordcloud
  image_colors = ImageColorGenerator(Mask)

  # Now we use the WordCloud function from the wordcloud library 
  wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(cluster)

  # Size of the image generated 
  plt.figure(figsize=(10,20))

  # Here we recolor the words from the dataset to the image's color
  # recolor just recolors the default colors to the image's blue color
  # interpolation is used to smooth the image generated 
  plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

  plt.axis('off')
  plt.show()

cluster_0_words = ' '.join(text for text in cluster_0['Tweet'])

Cluster 0: Tells about the problems related to services - !! Different clusers are formed each time!!
wordcloud(cluster_0_words)
cluster_1=frame[frame['Cluster'] == 1]
cluster_1
cluster_1_words = ' '.join(text for text in cluster_1['Tweet'])

cluster 1: tells about the problems about renatal amount
wordcloud(cluster_1_words)
The topics in one of the clusters is all over the place. Let us create a more detailed clusters

num_clusters = 8
km = KMeans(n_clusters=num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()
tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
frame = pd.DataFrame(tweets, index = [clusters])
frame.head()
frame['Cluster'].value_counts()
cluster_0=frame[frame['Cluster'] == 0]
cluster_0
cluster_0=frame[frame['Cluster'] == 0]
cluster_0_words = ' '.join(text for text in cluster_0['Tweet'])
wordcloud(cluster_0_words)
cluster_1=frame[frame['Cluster'] == 1]
cluster_1_words = ' '.join(text for text in cluster_1['Tweet'])
wordcloud(cluster_1_words)
cluster_2=frame[frame['Cluster'] == 2]
frame[frame['Cluster'] == 2]

cluster_2_words = ' '.join(text for text in cluster_2['Tweet'])
wordcloud(cluster_2_words)
cluster_3=frame[frame['Cluster'] == 3]
frame[frame['Cluster'] == 3]
cluster_3_words = ' '.join(text for text in cluster_3['Tweet'])

wordcloud(cluster_3_words)
cluster_4=frame[frame['Cluster'] == 4]
frame[frame['Cluster'] == 4]
cluster_4_words = ' '.join(text for text in cluster_4['Tweet'])

wordcloud(cluster_4_words)
cluster_5=frame[frame['Cluster'] == 5]
frame[frame['Cluster'] == 5].head()
cluster_5_words = ' '.join(text for text in cluster_5['Tweet'])

wordcloud(cluster_5_words)
cluster_6=frame[frame['Cluster'] == 6]
frame[frame['Cluster'] == 6]

cluster_6_words = ' '.join(text for text in cluster_6['Tweet'])
wordcloud(cluster_6_words)
cluster_7=frame[frame['Cluster'] == 7]

frame[frame['Cluster'] == 7]
cluster_7_words = ' '.join(text for text in cluster_7['Tweet'])
wordcloud(cluster_7_words)
frame.to_csv('clustered_tweets.csv') 

def identify_topics(df, desc_matrix, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit(desc_matrix)
    clusters = km.labels_.tolist()
    tweets = {'Tweet': df["Clean_text"].tolist(), 'Cluster': clusters}
    frame = pd.DataFrame(tweets, index = [clusters])
    print(frame['Cluster'].value_counts())

    
    for cluster in range(num_clusters):
        cluster_words = ' '.join(text for text in frame[frame['Cluster'] == cluster]['Tweet'])
        wordcloud(cluster_words)
        



identify_topics(df, desc_matrix, 4)
4 is too less a number of clusters
identify_topics(df, desc_matrix, 6)
6 seems like a good number of clusters
frame.to_csv('clustered_tweets.csv') 
