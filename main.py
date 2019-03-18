import sqlite3
import preprocess.preprocess as prep
import model_selection.model_selection as ms

from sklearn.linear_model import Lasso

def main():
    conn = sqlite3.connect('./data/trasi.db')

    df = prep.main(conn, look_fwd=10, threshold=0.000, option_step=500.0)

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
    ms.main(df, models=models)


if __name__ == '__main__':
    main()
