import pandas as pd
import numpy as np
import random
from argparse import ArgumentParser
from pickle import dump
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer

stemmer =SnowballStemmer("english")
def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_and_stem(text):
    tokens = text.split()
    stems = stem_tokens(tokens)
    return stems

def prepare_data(filename):
    df = pd.read_excel(filename)
    df = df.sample(frac=1)
    train_data = df['preprocessedSA'].values.tolist()
    train_labels = df['sentimentScore'].values.tolist()
    return np.array(train_data), np.array(train_labels)

def cross_validate_grid_search(train_features, train_labels, n_jobs, verbose):
    parameters = {
        "gbc__n_estimators":[200, 250, 350],
        "gbc__max_depth":[4, 5, 6, 7, 8],
        "gbc__min_samples_split": [25, 27, 36, 54, 60],
        "gbc__min_samples_leaf": [5, 8, 10, 12],
        "gbc__learning_rate":[0.1, 0.01],
        "gbc__subsample":[0.8, 0.9, 1.0]#must be (0.0,1.0]
    }
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True)
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('gbc', GradientBoostingClassifier()),
    ])
    gs_results = GridSearchCV(pipeline, parameters, scoring=make_scorer(f1_score, average='weighted'), 
                        cv=2, n_jobs=n_jobs, verbose=verbose)
    gs_results.fit(train_features, train_labels)

    return gs_results

def cross_validate_random_search(train_features, train_labels, n_jobs, verbose, n_iter):
    parameters = {
        "gbc__n_estimators":list(range(100,501,50)),
        "gbc__max_depth":list(range(4,21)),
        "gbc__min_samples_split": [20, 25, 27, 36, 45, 54, 56, 60],
        "gbc__min_samples_leaf": list(range(5,31, 5)),
        "gbc__learning_rate":[0.1, 0.01, 0.05, 0.005],
        "gbc__subsample":[0.5, 0.6, 0.8, 0.9, 1.0]#must be (0.0,1.0]
    }
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True)
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('gbc', GradientBoostingClassifier()),
    ])
    rs_results = RandomizedSearchCV(pipeline, parameters, scoring=make_scorer(f1_score, average='weighted'), 
                        cv=2, n_jobs=n_jobs, n_iter=n_iter, verbose=verbose)
    rs_results.fit(train_features, train_labels)

    return rs_results

if __name__ == '__main__':
    parser = ArgumentParser(description='Sentiment Classification Gradient Boosting Classifier Cross Validation')
    parser.add_argument('--data', type=str,
                        help='Data File', default="train_data.xlsx")
    parser.add_argument('--output', type=str,
                        help='Output Directory', default="cv-results")
    parser.add_argument('--search', type=int,
                        help='Random Search:1 or Grid Search:2', default=1)    
    parser.add_argument('--verbose', type=int,
                        help='verbosity: >1:time & param;>2:1 + score;>3:1,2 + start time', default=3)
    parser.add_argument('--threads', type=int,
                        help='number of cpu threads', default=8)
    parser.add_argument('--iter', type=int,
                        help='number of iterations(for random search)', default=2000)    
    args = parser.parse_args()
    
    outPath = args.output
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    
    train_data, train_labels = prepare_data(args.data)
    if args.search==1:
        results = cross_validate_random_search(train_data, train_labels, args.threads, args.verbose, args.iter)
    elif args.search==2:
        results = cross_validate_grid_search(train_data, train_labels, args.threads, args.verbose)
    else:
        print("Wrong arg value")
        exit(1)
    
    df = pd.DataFrame(results.cv_results_)
    df.to_csv(os.path.join(outPath, 'cv_results.csv'), index=False)
    with open(os.path.join(outPath, 'cv_results.pkl'), 'wb')as f:
        dump(results, f)
    with open(os.path.join(outPath, 'best_params.txt'), 'w')as f:
        f.write(f'Best parameters are: {results.best_params_}')