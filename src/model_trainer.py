import joblib
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)  # Moves one level up
DATASET_PATH = os.path.join(project_root, 'data', 'diabetes_data.csv')

ALL_COLUMNS = [
    'age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
    'polyphagia', 'irritability', 'partial_paresis', 'class'
]
COLUMNS_FOR_TRAINING = ALL_COLUMNS[:-1]


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def create_model(path_to_save):
    X, y = _load_data(DATASET_PATH)
    model = _train_model(X, y)
    save_model(model, path_to_save)
    return model


def predict(model, data):
    df = pd.DataFrame([data])
    df = df[COLUMNS_FOR_TRAINING]
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    print(df.columns)
    result = model.predict(df)
    print(f'Result: {result}')
    return bool(model.predict(df)[0])


def _load_data(path):
    df = pd.read_csv(path, sep=';', usecols=ALL_COLUMNS)
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    X = df.drop(columns=['class'])
    y = df['class']
    return X, y


def _train_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=20, n_estimators=5)
    model.fit(X_train, y_train)
    return model
