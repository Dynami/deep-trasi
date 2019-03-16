import sqlite3
import preprocess.preprocess as prep
import model_selection.model_selection as ms


def main():
    conn = sqlite3.connect('./data/trasi.db')

    df = prep.main(conn, look_fwd=10, threshold=0.015, option_step=250.0)
    ms.main(df)


if __name__ == '__main__':
    main()
