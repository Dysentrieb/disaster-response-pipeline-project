import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from joblib import dump, load


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DataTable', engine)

# load 
model = load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #category_counts = df.groupby('category').count()['message']
    category_names = list(genre_counts.index)
    
    y = df.drop(['id', 'genre', 'message', 'original'], axis=1)
    #y_sum = pd.DataFrame(y.sum(), columns = ['count'])
    y_sum = y.sum()
    y_cols = y.columns
    y_cols_no_related = y.drop(['related'], axis=1).columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
 
    
    graph_one= []

    graph_one.append(
      Bar(
      x = genre_names,
      y = genre_counts,
      )
    )

    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre',),
                yaxis = dict(title = 'Count'),
                )
    
    graph_two= []

    graph_two.append(
      Bar(
      x = list(y_cols),
      y = list(y_sum),
      )
    )

    layout_two = dict(title = 'Distribution of Message Categories',
                xaxis = dict(title = 'Category',),
                yaxis = dict(title = 'Count'),
                )
    
    graph_three= []

    graph_three.append(
      Bar(
      x = list(y_cols_no_related),
      y = list(y_sum.drop('related')),
      )
    )

    layout_three = dict(title = 'Distribution of Message Categories except "related"',
                xaxis = dict(title = 'Category',),
                yaxis = dict(title = 'Count'),
                )
    
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()