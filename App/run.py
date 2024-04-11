import json
import pandas as pd
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize Flask application
app = Flask(__name__)

class DisasterResponseApp:
    def __init__(self, database_filepath, model_filepath):
        self.database_filepath = database_filepath
        self.model_filepath = model_filepath
        self.model = None
        self.df = None

    def load_data(self):
        engine = create_engine(f'sqlite:///{self.database_filepath}')
        self.df = pd.read_sql_table('disaster_messages', engine)

    def load_model(self):
        self.model = joblib.load(self.model_filepath)

    def preprocess_data(self, df):
        genre_counts = df.groupby('genre').count()['message']
        genre_names = list(genre_counts.index)
        category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum(axis=0)
        category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns.tolist()
        return genre_names, genre_counts, category_names, category_counts

    @staticmethod
    def tokenize(text):
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
        return clean_tokens

    def predict_classification(self, message):
        input_message = DisasterResponseApp.tokenize(message)  # Reference the static method correctly
        prediction = self.model.predict([input_message])[0]
        prediction_labels = list(self.df.columns[4:])
        classification_results = dict(zip(prediction_labels, prediction))
        return classification_results

    def run(self):
        self.load_data()
        self.load_model()
        self.app.run(host='0.0.0.0', port=3001, debug=True)

# Index webpage with visualizations and user input form
@app.route('/')
@app.route('/index')
def index():
    # Create app_instance and load data
    app_instance = DisasterResponseApp('../data/DisasterResponse.db', '../models/classifier.pkl')
    app_instance.load_data()

    # Load and preprocess data
    genre_names, genre_counts, category_names, category_counts = app_instance.preprocess_data(app_instance.df)

    # Create visualizations
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        }
    ]

    # Encode graphs to JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render webpage with visualizations
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Predict page
@app.route('/go', methods=['POST'])
def go():
    # Get user input message
    message = request.form['message']

    # Create app_instance and load model
    app_instance = DisasterResponseApp('../data/DisasterResponse.db', '../models/classifier.pkl')
    app_instance.load_model()

    # Make prediction
    classification_results = app_instance.predict_classification(message)

    # Render webpage with prediction results
    return render_template('go.html', query=message, classification_result=classification_results)

# Main function
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()