#Processing the data

import pandas as pd
import matplotlib.pyplot as plt
import os

import nltk
#nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer

data = pd.read_csv("C:\\Users\\leroy\\personal\\TextAnalyticsCase.csv")
data.columns

 data.rename(columns={'text': 'tweettext'}, inplace= True)
 data['tweettext'][2]

 #preparing the data
 #only lower case
 data['tweettext'] = data['tweettext'].apply(lambda x: " ".join(x.lower() for x in x.split()))
 data['tweettext'][2]

 #remove numerical values and punctuation from the words
 patterndigits = '\\b[0-9]+\\b'
 data['tweettext'] = data['tweettext'].str.replace(patterndigits,'')

 #remove punctuation
patternpunc = '[^\w\s]'
data['tweettext'] = data['tweettext'].str.replace(patternpunc,'')
data['tweettext'][2]

#remove stop words
stop = stopwords.words('english')
data['tweettext'] = data['tweettext'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


 #stem the words
 porstem = PorterStemmer()
 data['tweettext'] = data['tweettext'].apply(lambda x: " ".join([porstem.stem(word) for word in x.split()]))

  #document-term matrix
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
tokens_data = pd.DataFrame(vectorizer.fit_transform(data['tweettext']).toarray(), columns=vectorizer.get_feature_names())
tokens_data.columns

#top 10 terms
sort_text = tokens_data.sum()
sort_text.sort_values(ascending = False).head(10)


#Using TF-IDF

import pandas as pd
>>> import numpy as np
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> from sklearn.feature_extraction.text import TfidfVectorizer

count= CountVectorizer()
>>> word_count= count.fit_transform(data)
>>> print(word_count)

from numpy import array
print(word_count.toarray())

#IDF 
 tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count.get_feature_names(),columns=["idf_weights"])


#invert document frequency
df_idf.sort_values(by=['idf_weights'])


