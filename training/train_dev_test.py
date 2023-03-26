import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Callable


def train_dev_test_split(
    dataset: pd.DataFrame,
    test_size_1: float,
    test_size_2: float
) -> pd.Series:
    """
    Split data into train, development(dev)/validation and test sets.

    :param dataset: a DataFrame contained the preprocessed data.
    :param test_size_1: the test size proportion between train and (dev+test).
    :param test_size_2: the test size proportion between dev and test.
    :return: the proportion of  X(train, dev, test) and the corresponding y sets.
    """
    corpus = dataset['sentence']
    labels = dataset['language']

    # Split data into train and dev+test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        corpus, labels, test_size=test_size_1, random_state=42)

    # Split temp data into dev and test sets
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_size_2, random_state=42)

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def train_model(
        convert_features: Callable,
        model_settings: Callable,
        x_train: pd.Series,
        y_train: pd.Series
) -> object:
    """ 
    Transform the text into vector features and fit into model.

    :param convert_features: A function to transform text to vector features.
    :param model_settings: A ML model with optional parameters.
    :param x_train: A X_train examples proportional data using to train the model.
    :param y_train: A y_train examples proportional contains corresponding label of each line of train dataset.
    :return: a model resulting from the training.
    """

    model = Pipeline([
        ('features_conv', convert_features),
        ('model_name', model_settings)
    ])

    model.fit(x_train, y_train)

    return model


def compare_models(
        model_lists: list,
        analyzers: list,
        n_initial,
        n_final,
        step,
        x_tr: pd.Series,
        y_tr: pd.Series,
        x_dv: pd.Series,
        y_dv: pd.Series
) -> None:
    """ 
    Feed various models, train and evaluate with validation data to compare their performance.

    :param model_lists: a list of models to be compared.
    :param analyzers: a list of analyzers (see n_gram_range of TfidfVectorizer in sklearn) to be compared.
    :param n_initial: initial ngram.
    :param n_final + 1: final ngram to be trained and compared.
    :param step: step between ngrams.
    :param x_tr: X_train examples.
    :param y_tr: y_train examples.
    :param X_dv: X_dev examples.
    :param y_dv: X_dev examples.
    """
    for model_list in model_lists:
        print(f"Model: {model_list}")
        for analyzer in analyzers:
            print(f"Analyzer: {analyzer}")
            for n in range(n_initial, n_final+1, step):
                model_trial = train_model(TfidfVectorizer(
                    analyzer=analyzer, ngram_range=(n, n)), model_list, x_tr, y_tr)
                y_pred = model_trial.predict(x_dv)
                accuracy = accuracy_score(y_dv, y_pred)
                print(f"\tn_gram {n} --> accuracy: {accuracy: .4f}")


def evaluate_model(model: object, x: pd.Series, y: pd.Series) -> None:
    """
    Evaluate the model on development/validation or test sets.

    :param x: a X_dev or X_test proportional data using to evaluate the model.
    :param y: A y_dev or y_text proportional contains corresponding label of each line of dev dataset.
    """
    y_pred = model.predict(x)

    print("Accuracy: ", accuracy_score(y, y_pred))
    print("Confusion Matrix: ", confusion_matrix(y, y_pred))
    print("Classification Report: ", classification_report(y, y_pred))
