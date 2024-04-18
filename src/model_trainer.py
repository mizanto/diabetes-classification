import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency


DATASET_PATH = '../data/diabetes_data.csv'


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
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    return bool(model.predict(df)[0])


def _cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def _calculate_correlation(df, cat_variables, threshold):
    corr_w_target = {}

    for variable in cat_variables:
        corr_w_target[variable] = _cramers_v(df[variable], df['class'])

    return [v for v, corr in corr_w_target.items() if corr > threshold]


def _load_data(path):
    df = pd.read_csv(path, sep=';')
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    cat_var = ['gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
               'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
               'itching', 'irritability', 'delayed_healing', 'partial_paresis',
               'muscle_stiffness', 'alopecia', 'obesity']

    significant_variables = _calculate_correlation(df, cat_var, 0.1)
    X = df[['age'] + significant_variables]
    y = df['class']

    return X, y


def _train_model(X_train, y_train):
    search_space = [
        {
            'classifier': [LogisticRegression()],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['liblinear']
        },
        {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [5, 10, 20],
            'classifier__max_depth': [5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    ]

    pipeline = Pipeline(steps=[('classifier', LogisticRegression())])
    cv_model = GridSearchCV(estimator=pipeline,
                            param_grid=search_space,
                            cv=5,
                            verbose=1,
                            n_jobs=-1)
    cv_model.fit(X_train, y_train)
    return cv_model.best_estimator_


if __name__ == '__main__':
    X, y = _load_data('../data/diabetes_data.csv')
    model = _train_model(X, y)
    save_model(model, '../models/diabetes_model.pkl')
