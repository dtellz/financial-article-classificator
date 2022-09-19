from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import json
import pickle

financial_corpus_df = pd.read_csv('./dataset/training_data.csv')

financial_corpus_df['category'].unique()
print(financial_corpus_df['category'].unique())
#['International_Finance' 'Earning_Reports' 'Commodities' 'Economy' 'Fraud'
# 'Mergers_Acquisitions' 'Policy' 'Oil' 'Capital' 'Litigation'
# 'Real_Estate']

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(financial_corpus_df['category'])
# create a new column 'label' as a number id for categories
financial_corpus_df['label'] = label_encoder.transform(financial_corpus_df['category'])
# convert categories into numbers (category id)
print(financial_corpus_df['label'].unique())
# [ 5  2  1  3  4  7  9  8  0  6 10]

# remove low level information (noise) from the articles.
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

x = financial_corpus_df['body']
y = financial_corpus_df['label']

vectorized_x = vectorizer.fit_transform(x)

rf_clf = RandomForestClassifier()

rf_clf.fit(vectorized_x, y)

pickle.dump(rf_clf, open('financial_text_classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('financial_text_vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open('financial_text_encoder.pkl', 'wb'))
