import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


def main(df, split_ratio=0.80):
    n_train = int(df.shape[0] * split_ratio)

    x_train_data = df.iloc[:n_train, :-2]
    x_test_data = df.iloc[n_train:, :-2]

    y_train_data = df.iloc[:n_train, -2:]
    y_test_data = df.iloc[n_train:, -2:]

    print(x_train_data.shape, x_test_data.shape, y_train_data.shape, y_test_data.shape)

    y_labels = np.reshape(y_train_data[['y_data_sparse']].values, (-1))
    max_count = pd.Series(y_labels).value_counts().max()
    class_weight = (pd.Series(y_labels).value_counts() / max_count).to_dict()

    print(class_weight)
    best_parameters = {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}

    if (False):
        parameters = {'solver': ('liblinear', 'saga'), 'penalty': ['l1'], 'C': [0.1, 1, 10]}
        model = LogisticRegression(class_weight=class_weight, random_state=True)
        clf = GridSearchCV(model, parameters, cv=4)
        clf.fit(x_train_data.values, y_train_data[['y_data_sparse']].values)

        print(clf.best_params_)
        model = clf.best_estimator_

    model = LogisticRegression(**best_parameters)
    model.fit(x_train_data.values, y_train_data[['y_data_sparse']].values)
    y_pred = model.predict(x_test_data)

    y_pred_proba = model.predict_proba(x_test_data)
    y_pred_proba = [y_pred_proba[i, v] for i, v in enumerate(np.argmax(y_pred_proba, axis=1))]
    y_pred_proba = np.array(y_pred_proba)

    y_test_data_sparse = y_test_data[['y_data_sparse']].values
    y_test_data_sparse = np.reshape(y_test_data_sparse, (-1))
    y_test_data_return = y_test_data[['y_data']].values
    y_test_data_return = np.reshape(y_test_data_return, (-1))

    print(y_pred.shape, y_test_data_sparse.shape, y_pred_proba.shape, y_test_data_return.shape)
    result = y_pred * y_test_data_sparse * y_pred_proba * y_test_data_return
    print('>>>>>', np.sum(result), '#####', result)
    pass


if __name__ == '__main__':
    df = pd.read_csv('../trasi.csv', index_col='index')
    main(df)
