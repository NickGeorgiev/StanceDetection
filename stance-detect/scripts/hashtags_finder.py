import json

from collections import Counter

from preprocessor import get_hashtags

data = []
with open('../jsons/labeled_data.json', 'r') as file:
    data = json.load(file)

all_hastags = [hashtag for elem in (get_hashtags(tweet)
                            for tweet, target, stance in data if stance == 'AGAINST') 
                                for hashtag in elem]

hashtags_counts = Counter(all_hastags).most_common(10)

with open('../jsons/common_hashtags.json', 'w') as file:
    file.write(json.dumps([hashtag for hashtag, _ in hashtags_counts]))