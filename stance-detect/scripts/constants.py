url_regex = r'(ftp|http(s)?)://.*?(?=\s|$)'
hashtag_symbol = '#'
hashtag_regex = r"#(\w+)"
user_mention_regex = r'@\w+\s?'
capitalized_text_regex = r'\b[A-Z\s]+\b'
feature_punctuation = r'\!|\?|\.{2,}'
quotes = r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)"
verb_pronoun_regex = r"((VB|VBD|VBG|VBN|VBP|VBZ)\s(PRP|PRP$))"


contractions = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\byou're\b": "you are",
    r"\bwe're\b": "we are",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcnt\b": "can not",
    r"\bcannot\b": "can not",
    r"\bm\b": "am",
    r"\bIm\b": "I am",
    r"\byoure\b": "you are",
    r"\bdont\b": "do not",
    r"\bdoesnt\b": "does not",
    r"\bdidnt\b": "did not",
    r"\bhasnt\b": "has not",
    r"\bhavent\b": "have not",
    r"\bhadnt\b": "had not",
    r"\bwouldnt\b": "would not",
    r"\bcant\b": "can not",
    r"\btheyre\b": "they are",
    r"\bthey're\b": "they are",
    r"\bll\b": "will",
}