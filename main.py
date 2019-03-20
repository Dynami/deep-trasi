import sqlite3
import preprocess.preprocess as prep
import model_selection.model_selection as ms
import process.execution as exec
from sklearn.linear_model import Lasso
import pandas as pd

def main(load_computed_data=True, look_fwd=10):
    conn = sqlite3.connect('./data/trasi.db')

    if load_computed_data:
        df = pd.read_csv('./trasi.csv', index_col='index')
        df.drop(columns='Unnamed: 0', inplace=True)
    else:
        df = prep.main(conn, look_fwd=look_fwd, threshold=0.000, option_step=500.0)

    models =[
        {
            # Total >>>>> 263.927 %
            # Prec. >>>>> 65.854 %
            # Cover.>>>>> 100.000 %
            # Sharp >>>>> 0.400
            # Mean  >>>>> 0.021
            # Std   >>>>> 0.054
            'clf': Lasso,
            'use_class_weight': False,
            'parameters': {
                'alpha': [1.0],  # np.arange(1.6, 2.2, 0.2),
                'selection': ['cyclic'],
                'random_state': [True]
            }
        }
    ]
    model = ms.main(df, models=models)

    x_train, x_test, y_train, y_test = ms.split_dataset(df, split_ratio=0.8)
    exec.main(model, x_test=x_test, y_test=y_test, look_fwd=look_fwd)

if __name__ == '__main__':
    main(load_computed_data=True)
