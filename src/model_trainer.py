import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import cramers_v


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def _calculate_correlation(df, cat_variables, threshold):
    corr_w_target = {}

    for variable in cat_variables:
        corr_w_target[variable] = cramers_v(df[variable], df['class'])

    return [v for v, corr in corr_w_target.items() if corr > threshold]


def _load_data(path):
    df = pd.read_csv('../data/diabetes_data.csv', sep=';')
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    cat_var = ['gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
               'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
               'itching', 'irritability', 'delayed_healing', 'partial_paresis',
               'muscle_stiffness', 'alopecia', 'obesity']

    significant_variables = _calculate_correlation(df, cat_var, 0.1)
    X = df[significant_variables]
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
