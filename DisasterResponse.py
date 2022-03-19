# import libraries
import pandas as pd
import numpy as np
import re
import pickle
import nltk
import sqlite3

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class DisasterResponse:
    def __init(self, data_file, categories):
        self.data_file = data_file

        self.categories = categories

    def extract_data(self):
        """
        Reads the data into a Pandas DataFrame
        :return: DataFrame df
        """
        df = pd.read_csv(self.data_file)

        return df

    def transform_data(self, df):
        """
        Transforms the data frame
        :param df:
        :return df:
        """
        # load categories dataset
        categories = pd.read_csv(self.categories)

        # merge the categories into the data df
        df = df.merge(categories, how='left', on=['id'])

        # create a dataframe of the 36 individual category columns
        categories = df['categories'].str.split(';', expand=True)

        # extract a list of column names
        row = categories.iloc[0]

        category_column_names = row.str.split('-', expand=True)[0]

        # set the df columns to the category column names
        categories.columns = category_column_names

        # convert category columns to 1's and 0's
        for column in categories:
            # set each value to be the last character of the string
            categories[column] = categories[column].str[-1]

            # convert column from string to numeric
            categories[column] = categories[column].astype(int)

        # set the data df categories columns to the cleaned data
        df.drop(['categories'], axis=1, inplace=True)

        df = pd.concat([df, categories], axis=1)

        # Remove any duplicates
        df.drop_duplicates(inplace=True)

        return df

    def load_data(self, df):
        """
        Loads data from the data frame into the database
        :return: pd.DataFrame df
        """
        # Setup connection
        conn = sqlite3.connect('DisasterResponse.db')

        # get a cursor
        cur = conn.cursor()

        # Drop the table if it exists
        cur.execute("DROP TABLE IF EXISTS categorized_messages")

        insert_sql = 'CREATE TABLE categorized_messages'

        # create the table including id as a primary key
        cur.execute(
            'CREATE TABLE categorized_messages (id TEXT PRIMARY KEY, message TEXT, original TEXT, genre TEXT, related TEXT, request TEXT, offer TEXT, aid_related TEXT, medical_help TEXT, medical_products TEXT, search_and_rescue TEXT, security TEXT, military TEXT, child_alone TEXT, water TEXT, food TEXT, shelter TEXT, clothing TEXT, money TEXT, missing_people TEXT, refugees TEXT, death TEXT, other_aid TEXT, infrastructure_related TEXT, transport TEXT, buildings TEXT, electricity TEXT, tools TEXT, hospitals TEXT, shops TEXT, aid_centers TEXT, other_infrastructure TEXT, weather_related TEXT, floods TEXT, storm TEXT, fire TEXT, earthquake TEXT, cold TEXT, other_weather TEXT, direct_report TEXT);')

        # insert data
        df.to_sql('categorized_messages', con=conn, if_exists='replace')

        # commit changes made to the database
        conn.commit()

        # select all from the categorized_messages table
        cur.execute("SELECT * FROM categorized_messages")

        results = cur.fetchall()

        results


    def build_model(self):
        # text processing and model pipeline


        # define parameters for GridSearchCV


        # create gridsearch object and return as final model pipeline


        return model_pipeline


    def train(self, X, y, model):
        # train test split


        # fit model


        # output model test results


        return model


    def export_model(self, model):
        # Export model as a pickle file



    def run_pipeline():
        df = load_data()  # run ETL pipeline
        data = clean_data(df)
        model = build_model()  # build model pipeline
        model = train(X, y, model)  # train model pipeline
        export_model(model)  # save model
