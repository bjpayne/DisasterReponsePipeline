import re

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


def tokenize(text):
    """
    Tokenizer for the ML pipeline
    :param text:
    :return:
    """
    # Normalize text
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', text)

    # Fetch tokens
    tokens = []

    lemmatizer = WordNetLemmatizer()

    stemmer = PorterStemmer()

    for token in word_tokenize(text):
        lemmatized = lemmatizer.lemmatize(token)

        stemmed = stemmer.stem(lemmatized)

        tokens.append(stemmed)

    return tokens