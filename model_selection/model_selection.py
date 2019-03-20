import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import warnings

warnings.filterwarnings("ignore")

def custom_scorer(y_true, y_pred):
    return np.sum(y_pred * y_true)

def split_dataset(df, split_ratio):
    # df = df[::-1]
    n_train = int(df.shape[0] * split_ratio)

    x_train_data = df.iloc[:n_train, :-2]
    x_test_data = df.iloc[n_train:, :-2]

    y_train_data = df.iloc[:n_train, -2:]
    y_test_data = df.iloc[n_train:, -2:]

    print(x_train_data.shape, x_test_data.shape, y_train_data.shape, y_test_data.shape)

    return x_train_data, x_test_data, y_train_data, y_test_data


def get_class_weight(y_train):
    y_labels = np.reshape(y_train[['sparse_target']].values, (-1))
    max_count = pd.Series(y_labels).value_counts().max()
    return (pd.Series(y_labels).value_counts() / max_count).to_dict()


def evaluate(clf, x_train, y_train, class_weight=None, cv=5, scoring='accuracy'):

    if class_weight is not None:
        model = clf['clf'](class_weight=class_weight, random_state=True)
    else:
        model = clf['clf']()

    score_fn = make_scorer(custom_scorer)

    print(type(model))
    np.random.seed(2888306641)  # Best seed found
    seed = np.random.get_state()[1][0]
    print('np.random.seed', seed)
    clf = GridSearchCV(model, clf['parameters'], cv=cv, scoring=score_fn)
    clf.fit(x_train, y_train)

    print(clf.best_params_)

    return {
        'model':clf.best_estimator_,
        'best_params': clf.best_params_,
        'best_score': clf.best_score_
    }


def main(df, split_ratio=0.80, models=None):

    x_train_data, x_test_data, y_train_data, y_test_data = split_dataset(df, split_ratio)

    class_weight = get_class_weight(y_train_data)
    print('class_weight', class_weight)
    clfs = []

    for i, clf in enumerate(models):
        _class_weight = class_weight if clf['use_class_weight'] else None
        _clf = evaluate(clf, x_train=x_train_data.values, y_train=y_train_data[['sparse_target']].values, class_weight=_class_weight)

        #_result = scoring(_clf['model'], x_test_data.values, y_test_data)
        #_clf['results'] = _result
        clfs.append(_clf)
        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')

    #print(clfs)

    clfs = sorted(clfs, key= lambda x: x['best_score'])
    print('-----------------------------------------------')
    print(clfs[-1])
    return clfs[-1]


if __name__ == '__main__':
    df = pd.read_csv('../trasi.csv', index_col='index')
    df.drop(columns='Unnamed: 0', inplace=True)
    from sklearn.linear_model import LogisticRegression, Lasso, LassoLars, Lars
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    models = [
        # {
        #     'clf':LogisticRegression,
        #     'use_class_weight':True,
        #     'parameters': [
        #         {
        #             'solver':['liblinear', 'saga'],
        #             'penalty':['l1', 'l2'],
        #             'C': [0.1, 1, 10]
        #         },
        #         {
        #             'solver': ['newton-cg', 'lbfgs', 'sag'],
        #             'penalty': ['l2'],
        #             'C': [0.1, 1, 10]
        #         }
        #     ]
        # },
        # {
        #
        #     # Total >>>>> 15.118%
        #     # Prec. >>>>> 57.778%
        #     # Cover.>>>>> 73.171%
        #     # Sharp >>>>> 0.046
        #     # Mean  >>>>> 0.002
        #     # Std   >>>>> 0.036
        #     'clf': SVC,
        #     'use_class_weight':True,
        #     'parameters': {
        #         'probability': [True],
        #         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], #, 'precomputed'
        #         'C': [0.1, 1, 10],
        #         'gamma': ['auto', 0.1, 1., 2.]
        #     }
        # },
        # {
        #     # Total >>>>> 37.302 %
        #     # Prec. >>>>> 64.912 %
        #     # Cover.>>>>> 46.341 %
        #     # Sharp >>>>> 0.196
        #     # Mean  >>>>> 0.007
        #     # Std   >>>>> 0.033
        #     'clf': DecisionTreeClassifier,
        #     'use_class_weight':True,
        #     'parameters':{
        #         'criterion':['gini', 'entropy'],
        #         'splitter': ['best', 'random']
        #     }
        # },
        # {
        #     # Total >>>>> -11.002 %
        #     # Prec. >>>>> 58.889 %
        #     # Cover.>>>>> 73.171 %
        #     # Sharp >>>>> -0.033
        #     # Mean  >>>>> -0.001
        #     # Std   >>>>> 0.038
        #     'clf':RandomForestClassifier,
        #     'use_class_weight':True,
        #     'parameters':{
        #         'criterion':['gini', 'entropy'],
        #         'max_depth': [2, 3, 5, None],
        #         'bootstrap':[True, False]
        #     }
        # },
        # {
        #     # Total  >>>>> -22.661 %
        #     # Prec.  >>>>> 46.835 %
        #     # Cover. >>>>> 64.228 %
        #     # Sharp  >>>>> -0.080
        #     # Mean   >>>>> -0.003
        #     # Std    >>>>> 0.036
        #     'clf': AdaBoostClassifier,
        #     'use_class_weight': False,
        #     'parameters': {
        #         'algorithm': ['SAMME', 'SAMME.R'],
        #         'n_estimators': [30, 50, 70],
        #         'learning_rate': [0.5, 1.0, 2.0]
        #     }
        # },
        # {
        #     # Total >>>>> 5.032 %
        #     # Prec. >>>>> 57.143 %
        #     # Cover.>>>>> 56.911 %
        #     # Sharp >>>>> 0.017
        #     # Mean  >>>>> 0.001
        #     # Std   >>>>> 0.042
        #     'clf':GradientBoostingClassifier,
        #     'use_class_weight': False,
        #     'parameters': {
        #         'loss': ['deviance'],
        #         'learning_rate':[0.01, 0.1, 1],
        #         'n_estimators': [50, 100]
        #     }
        #},
        # {
        #     # Total  >>>>> 70.821 %
        #     # Prec.  >>>>> 64.557 %
        #     # Cover. >>>>> 64.228 %
        #     # Sharp  >>>>> 0.314
        #     # Mean   >>>>> 0.009
        #     # Std    >>>>> 0.029
        #     'clf': MLPClassifier,
        #     'use_class_weight': False,
        #     'parameters': {
        #         'shuffle': [True],
        #         'hidden_layer_sizes':[(200)],
        #         'activation': ['relu'],
        #         #'nesterovs_momentum': [True, False],
        #         'solver': ['adam'],
        #         'alpha':[0.0001],
        #         'learning_rate': ['invscaling'],# 'constant', 'adaptive',
        #         #'early_stopping':[True],
        #     }
        # },
        {
            #Total >>>>> 263.927 %
            #Prec. >>>>> 65.854 %
            #Cover.>>>>> 100.000 %
            #Sharp >>>>> 0.400
            #Mean  >>>>> 0.021
            #Std   >>>>> 0.054
            'clf': Lasso,
            'use_class_weight': False,
            'parameters': {
                'alpha': [1.0], # np.arange(1.6, 2.2, 0.2),
                'selection': ['cyclic'],
                'random_state':[True]
            }
        },
        # {
        #     # Total >>>>> -3.855 %
        #     # Prec. >>>>> 52.273 %
        #     # Cover.>>>>> 71.545 %
        #     # Sharp >>>>> -0.012
        #     # Mean  >>>>> -0.000
        #     # Std   >>>>> 0.036
        #     'clf': XGBClassifier,
        #     'use_class_weight': False,
        #     'parameters': {
        #         'booster': ['gbtree', 'gblinear', 'dart'],
        #         'reg_alpha': [0.1, 1.0, 2.],
        #         'reg_lambda': [0.001]
        #     }
        # }
    ]

    main(df, models=models)
