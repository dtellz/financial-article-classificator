from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open('./model/financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('./model/financial_text_vectorizer.pkl', 'rb')) 
label_encoder = pickle.load(open('./model/financial_text_encoder.pkl', 'rb'))

def process(inPath, outPath):
    #read inpy file
    input_df = pd.read_csv(inPath)
    #vectorize the data
    feature = tfidf_vectorizer.transform(input_df['body'])
    # predict the classes
    prediction = model.predict(features)
    output_df = input_df[['id', 'category']]
    output_df.to_csc(outPath, index=False)

grav.wait_for_requests(process)