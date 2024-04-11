# Required Libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump

# Downloading NLTK resources
nltk.download(['punkt', 'wordnet'])

# Function to load data
def load_data(database_filepath):
    """Load data from SQLite database"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y

# Tokenization and Lemmatization
def tokenize(text):
    """Tokenize and lemmatize text"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Building model pipeline
def build_model():
    """Build classifier pipeline and tune model using GridSearchCV"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv

# Train the model
def train_model(model, X_train, Y_train):
    """Train the model with training data"""
    model.fit(X_train, Y_train)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, Y_test):
    """Evaluate model performance and return classification report"""
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

# Save model
def save_model(model, model_filepath):
    """Export the final model using joblib"""
    dump(model, model_filepath)
    print(f"Model saved to: {model_filepath}")

# Main function
def main():
    """Main function to load, train, evaluate, and save the model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\nDATABASE:', database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        print('Training model...')
        model = train_model(model, X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        print('Saving model...\nMODEL:', model_filepath)
        save_model(model, model_filepath)
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.\n\n'
              'Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()