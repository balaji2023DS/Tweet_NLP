import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle

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

# Function to load the dataset
def load_dataset():
    return pd.read_csv(r'D:\\Guvi\\Final_Project\\FinalBalancedDataset.csv')

# Function to convert text using different NLP methods
def convert_text(df, method):
    df['processed_tweet'] = df['Tweet'].apply(preprocess_text)
    text = df['processed_tweet']

    if method == 'Bag of Words':
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(text)

    elif method == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(text)

    else:
        raise ValueError("Invalid NLP method selected.")
    
    # Apply TruncatedSVD to the full matrix
    svd = TruncatedSVD(n_components=100)
    X_svd = svd.fit_transform(X)
    return X_svd


# Function to train and evaluate a classifier
def train_evaluate_classifier(classifier, reducer, X_train, y_train, X_test, y_test):
    clf = classifier
    
    # For classifiers that support probability estimation, fit and transform the reduced data
    if hasattr(clf, "predict_proba"):
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
    else:
        X_train_reduced = X_train
        X_test_reduced = X_test
    
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_test_reduced)

    # Calculate metrics
    precision, recall, f1_score, _ = classification_report(y_test, y_pred, output_dict=True)['1'].values()
    conf_matrix = confusion_matrix(y_test, y_pred)

    # For classifiers that support probability estimation, calculate probabilities
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_reduced)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = None

    return precision, recall, f1_score, conf_matrix, roc_auc


# Main function
def main():

    # Step 1: Load the dataset
    df = load_dataset()
    df = df.sample(n=1000,random_state=42).reset_index(drop=True)
    X = df['Tweet']
    y = df['Toxicity']

    # Step 2: Convert text using Bag of Words and TF-IDF
    nlp_methods = ['Bag of Words', 'TF-IDF']
    for nlp_method in nlp_methods:
        X_nlp = convert_text(df, nlp_method)

        # Step 3: Train and evaluate classifiers
        classifiers = {
            'Decision Trees': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),            
            'K-NN Classifier': KNeighborsClassifier(),
            'SVM': SVC(probability=True)
        }

        reducers = {
            'Bag of Words': TruncatedSVD(n_components=100),
            'TF-IDF': TruncatedSVD(n_components=100)
        }

        for classifier_name, classifier in classifiers.items():
            for reducer_name, reducer in reducers.items():
                if (nlp_method == 'Bag of Words' and reducer_name == 'Bag of Words') or (nlp_method == 'TF-IDF' and reducer_name == 'TF-IDF'):
                    X_train, X_test, y_train, y_test = train_test_split(X_nlp, y, test_size=0.2, random_state=42)
                    X_train_reduced = reducer.fit_transform(X_train)
                    X_test_reduced = reducer.transform(X_test)
                    precision, recall, f1_score, conf_matrix, roc_auc = train_evaluate_classifier(classifier, reducer, X_train_reduced, y_train, X_test_reduced, y_test)

                    # Step 4: Print metrics for each classifier and NLP method
                    print(f"Classifier: {classifier_name}, NLP Method: {nlp_method}")
                    print(f"Precision: {precision:.2f}")
                    print(f"Recall: {recall:.2f}")
                    print(f"F1-Score: {f1_score:.2f}")
                    print("Confusion Matrix:")
                    print(conf_matrix)
                    print(f"ROC-AUC: {roc_auc:.2f}")

                    # Step 5: Save the trained model and NLP representation using pickle files
                    model_filename = f"{classifier_name}_{nlp_method}_model.pkl"
                    with open(model_filename, 'wb') as file:
                        if nlp_method == 'Bag of Words':
                                vectorizer = CountVectorizer()
                        elif nlp_method == 'TF-IDF':
                                vectorizer = TfidfVectorizer()

                        X_text = df['Tweet']
                        X_vectorized = vectorizer.fit_transform(X_text)
                        X_reduced = reducer.fit_transform(X_vectorized)
                        pickle.dump((classifier, reducer, vectorizer), file)


if __name__ == "__main__":
    main()
