import json
import plotly
import sqlite3

import joblib

import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie

from models.tokenizer import tokenize

app = Flask(__name__)

print(app.instance_path)

# load data
conn = sqlite3.connect('../data/DisasterResponse.sqlite')
df = pd.read_sql('SELECT * FROM categorized_messages', conn)

# load model
model = joblib.load("../models/model.sav")


# index webpage displays the graphs receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    categories = df[['related', 'request',
                     'offer', 'aid_related', 'medical_help', 'medical_products',
                     'search_and_rescue', 'security', 'military', 'child_alone', 'water',
                     'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
                     'death', 'other_aid', 'infrastructure_related', 'transport',
                     'buildings', 'electricity', 'tools', 'hospitals', 'shops',
                     'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
                     'storm', 'fire', 'earthquake', 'cold', 'other_weather',
                     'direct_report']].copy()
    category_names = list(categories.columns)

    # create graphs
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=categories.sum()
                )
            ],

            'layout': {
                'title': 'Messages per category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    values=categories.sum(),
                    labels=category_names
                )
            ],

            'layout': {
                'title': 'Distribution of messages',
                'height': 800,
                'width': 1200,
            }
        }
    ]
    
    # encode plotly graphs in JSON
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3000, debug=True)


if __name__ == '__main__':
    main()
    