import re
import string

from constants import *

punctuation_pattern = re.compile('[%s]' % re.escape(string.punctuation))


def initial_text_clean_up(text):
    text = text.encode('ascii', 'ignore').decode('unicode-escape')
    text = remove_escaped_characters(text)
    text = remove_urls(text)
    text = remove_user_mentions(text)
    text = replace_contractions(text)
    text = remove_unnecessary_whitespace(text)
    text = remove_numeric_data(text)
    return text


def replace_contractions(text):
    for key in contractions:
        text = re.sub(key, contractions[key], text, flags=re.IGNORECASE)
    return text


def remove_escaped_characters(text):
    text = text.replace('\\', '')
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    return text


def remove_unicode_characters(text):
    text = text.encode('ascii', 'ignore')
    return text.decode('utf-8')


def remove_urls(text):
    return re.sub(url_regex, '', text, flags=re.IGNORECASE)


def remove_unnecessary_whitespace(text):
    words = text.split()
    text = ' '.join(words)
    return text


def remove_hashtag_symbol(text):
    return re.sub(hashtag_symbol, '', text)


def remove_user_mentions(text):
    return re.sub(user_mention_regex, '', text)


def remove_hashtags(text):
    return re.sub(hashtag_regex, '', text)


def remove_punctuation(text):
    return re.sub(punctuation_pattern, '', text)


def get_hashtags(text):
    return re.findall(hashtag_regex, text)


def get_capitalized_text(text):
    return [e for e in re.findall(capitalized_text_regex, text) if not e.isspace() and e != '']


def get_punctuation(text):
    return re.findall(feature_punctuation, text)


def get_quoted_text(text):
    return re.findall(quotes, text)


def get_interjection_words_descriptions(text):
    text = remove_punctuation(text).lower()
    interjection_words_descriptions = []
    for word in text.split():
        if word in interjection_words:
            for word_description in interjection_words[word].split():
                interjection_words_descriptions.append(word_description)
    return interjection_words_descriptions


def remove_numeric_data(tweet):
    tweet = tweet.lower()
    words = tweet.split()
    for word in words:
        if word.encode('unicode-escape').startswith(b'\u') or word.isnumeric():
            words.remove(word)
    words = [word.strip(r'\"\'') for word in words]
    tweet = ' '.join(words).strip()
    return tweet


def get_tweet_text_only(tweet):
    tweet = remove_hashtags(tweet)
    tweet = remove_punctuation(tweet)

    tweet = tweet.lower()
    words = tweet.split()
    for word in words:
        if word.encode('unicode-escape').startswith(b'\u') or word.isnumeric():
            words.remove(word)
    # remove any quotes
    words = [word.strip(r'\"\'') for word in words]
    tweet = ' '.join(words).strip()
    return tweet
