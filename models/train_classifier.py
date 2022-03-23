# import libraries
import re
import pickle
import nltk
import sqlite3
import sys

import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
    Load the data into the script
    :param database_filepath:
    :return X, y, category_names:
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM categorized_messages', conn)

    X = df['message']
    y = df.drop(['index', 'id', 'message', 'original', 'genre'], axis=1).copy()

    y.fillna(0)

    category_names = y.columns

    return X, y, category_names


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

    english_stopwords = stopwords.words("english")

    lemmatizer = WordNetLemmatizer()

    stemmer = PorterStemmer()

    for token in word_tokenize(text):
        if token not in english_stopwords:
            lemmatized = lemmatizer.lemmatize(token)

            stemmed = stemmer.stem(lemmatized)

            tokens.append(stemmed)

    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'vect__max_df': [0.5],
        'clf__estimator__n_estimators': [10]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=3, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print out a classification report
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Save the completed model for re-use
    :param model:
    :param model_filepath:
    :return:
    """
    filename = model_filepath

    pickle.dump(model, open(filename, 'wb'))

def main():
    if len(sys.argv) == 3:
        # Download nltk packages
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
