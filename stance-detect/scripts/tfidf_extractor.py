import json 
from sklearn.feature_extraction.text import TfidfVectorizer


labeled_data = []
with open('../jsons/labeled_data.json', 'r') as file:
    labeled_data = [ tweet for tweet, target, stance in list(json.load(file))]

def calculateTFIDF():
    vectorizer = TfidfVectorizer()
    new_data = vectorizer.fit_transform(labeled_data)
    test_data = vectorizer.inverse_transform(new_data)
    idf = vectorizer.idf_
    res = dict(zip(vectorizer.get_feature_names(), idf))
    print(res['zip'])

calculateTFIDF()