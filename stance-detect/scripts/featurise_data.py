import json

from extract_features import extract_features_of_tweet


def create_features_for_data():
    labeled_data=[]
    with open('../jsons/labeled_data.json', 'r') as file:
        labeled_data = list(json.load(file))
    featuresets = [(extract_features_of_tweet(tweet), cls) for (tweet, target, cls) in labeled_data]
    with open('../jsons/features_sets.json', 'w') as file:
        file.write(json.dumps(featuresets))

create_features_for_data()