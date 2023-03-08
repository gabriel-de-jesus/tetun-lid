import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Callable


def train_dev_test_split(dataset: pd.DataFrame, test_size_1: float, test_size_2: float) -> pd.Series:
    """ split data into train, development(dev)/validation and test sets.

    Args:
        dataset (DataFrame): A DataFrame contained the preprocessed data.
        test_size_1 (float): the test size proportion between train and (dev+test).
        test_size_2 (float): the test size proportion between dev and test.

    Returns:
        The proportion of  X(train, dev, test) and the corresponding y sets.
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
        x_train: list,
        y_train: list
) -> object:
    """ Transform the text into vector features and fit into model.

    Args:
        convert_features (function): A function to transform text to vector features.
        model_settings (function): A ML model with optional parameters.
        x_train (series): A X_train proportional data using to train the model.
        y_train (series): A y_train proportional contains corresponding label of each line of train dataset.

    Returns:
        A model resulting from the training.
    """
    # define pipeline
    model = Pipeline([
        ('features_conv', convert_features),
        ('model_name', model_settings)
    ])

    # fit the model on the training data
    model.fit(x_train, y_train)

    return model


def evaluate_model(model: object, x: pd.Series, y: pd.Series) -> None:
    """ Evaluate the model on development/validation or test sets.

    Args:
        x (series): A X_dev or X_test proportional data using to evaluate the model.
        y (series): A y_dev or y_text proportional contains corresponding label of each line of dev dataset.
    """
    y_pred = model.predict(x)

    print("Accuracy: ", accuracy_score(y, y_pred))
    print("Confusion Matrix: ", confusion_matrix(y, y_pred))
    print("Classification Report: ", classification_report(y, y_pred))
