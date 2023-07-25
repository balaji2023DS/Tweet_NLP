from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

app = Flask(__name__)

# Load the pickle files for classifiers and reducers
classifiers = {
    'Decision Trees': 'Decision Trees',
    'Random Forest': 'Random Forest',
    'K-NN Classifier': 'K-NN Classifier',
    'SVM': 'SVM'
}

reducers = {
    'Bag of Words': 'Bag of Words',
    'TF-IDF': 'TF-IDF'
}

# Function to preprocess text
# Function for text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase and tokenize
    words = text.lower().split()
    # Remove stopwords and perform stemming and lemmatization
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    return " ".join(words)

# Function to load the classifier, reducer, and vectorizer from pickle files
def load_model(classifier_name, nlp_method):
    model_filename = f"{classifiers[classifier_name]}_{reducers[nlp_method]}_model.pkl"
    with open(model_filename, 'rb') as file:
        classifier, reducer, vectorizer = pickle.load(file)
    return classifier, reducer, vectorizer

# Function to predict toxicity
def predict_toxicity(text, classifier, reducer, vectorizer):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([preprocessed_text])

    # Reduce the dimensionality using the same reducer used during training
    text_reduced = reducer.transform(text_vectorized)

    # Make predictions using the classifier
    prediction = classifier.predict(text_reduced)[0]

    return prediction

# Routes for home and result pages
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    text = request.form['tweet']
    classifier_name = request.form['classifier']
    nlp_method = request.form['nlp_method']

    classifier, reducer, vectorizer = load_model(classifier_name, nlp_method)
    prediction = predict_toxicity(text, classifier, reducer, vectorizer)

    return render_template('result.html', tweet=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
