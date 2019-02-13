import json

from collections import Counter

from preprocessor import get_hashtags

data = []
with open('../jsons/labeled_data.json', 'r') as file:
    data = json.load(file)

all_hashtags_hillary = [hashtag for elem in (get_hashtags(tweet)
                            for tweet, target, stance in data if target == 'Hillary Clinton') 
                                for hashtag in elem]
                            
all_hashtags_climate = [hashtag for elem in (get_hashtags(tweet)
                            for tweet, target, stance in data if target == 'Climate Change is a Real Concern') 
                                for hashtag in elem]

all_hashtags_atheism = [hashtag for elem in (get_hashtags(tweet)
                            for tweet, target, stance in data if target == 'Atheism') 
                                for hashtag in elem]

all_hashtags_feminist = [hashtag for elem in (get_hashtags(tweet)
                            for tweet, target, stance in data if target == 'Feminist Movement') 
                                for hashtag in elem]

all_hashtags_abortion = [hashtag for elem in (get_hashtags(tweet)
                            for tweet, target, stance in data if target == 'Legalization of Abortion') 
                                for hashtag in elem]                               


hashtags_counts_hillary = Counter(all_hashtags_hillary).most_common(10)
hashtags_counts_climate = Counter(all_hashtags_climate).most_common(10)
hashtags_counts_atheism = Counter(all_hashtags_atheism).most_common(10)
hashtags_counts_feminist = Counter(all_hashtags_feminist).most_common(10)
hashtags_counts_abortion = Counter(all_hashtags_abortion).most_common(10)

res = {
    'Hillary': hashtags_counts_hillary,
    'Climate': hashtags_counts_climate,
    'Atheism': hashtags_counts_atheism,
    'Feminism': hashtags_counts_feminist,
    'Abortion': hashtags_counts_abortion
}

with open('../jsons/common_hashtags_.json', 'w') as file:
    file.write(json.dumps(res))