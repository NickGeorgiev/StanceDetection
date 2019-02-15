import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def display_topics(model, feature_names, top_words_number):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {0}: ".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-top_words_number - 1 :-1]]))


def get_topic_as_keywords(model, feature_names, top_words_number):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-top_words_number - 1 :-1]]))
    return topics


labeled_data = []
with open('../jsons/labeled_data.json', 'r') as file:
    labeled_data = list(json.load(file))

dataset = [tweet for tweet, target, stance in labeled_data]

features_count = 1000

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=features_count, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(dataset)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

topics_count = 5

nmf = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

top_words_count = 20

def extract_topic_features(features, tweet):
    i = 0
    for topic_cluster in nmf.transform(tfidf_vectorizer.transform([tweet]))[0]:
        features['topic_cluster' + str(i)] = topic_cluster
        i += 1
