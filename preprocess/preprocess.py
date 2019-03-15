import warnings
import pandas as pd
import datetime as dt
import numpy as np
import sqlite3
import calendar
import math
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def option_expiration(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    return dt.date(date.year, date.month, day)

def weekofmonth(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    third = dt.date(date.year, date.month, day)
    return 3+math.ceil((date-third).days/7)

def get_options(conn, df_index, option_step=250.0, upper_bound_strike=30_000, lower_bound_strike=10_000, maturity_filter=1.):
    sql_options = "select strftime('%Y-%m-%d', o.expire/1000, 'unixepoch', 'localtime') as 'expire', " \
                  "strftime('%Y-%m-%d', ob.time/1000, 'unixepoch', 'localtime') as 'date', " \
                  "o.type, o.strike, ob.volume, ob.open_interest, ob.settlement, " \
                  "ob.volatility/100 as 'volatility' " \
                  "from options o " \
                  "inner join option_bars ob on o.ticker = ob.ticker " \
                  "order by ob.time, o.expire, o.strike, o.type"

    df_options = pd.read_sql(sql_options, con=conn, index_col=['date'])
    df_options.index = pd.to_datetime(df_options.index.values)
    
    df_options[['volume', 'open_interest']] = df_options[['volume', 'open_interest']].astype(float)
    df_options.sort_index(inplace=True)
    df_options['expire'] = pd.to_datetime(df_options['expire'])

    df_options['maturity'] = (df_options['expire'] - df_options.index)
    df_options['maturity'] = (df_options['maturity'] / np.timedelta64(1, 'D')).astype(int) / 365.
    df_options.drop(columns=['expire'], axis=1, inplace=True)
    
    print('Before:', df_options.shape)
    df_options = df_options[(df_options['strike'] >= lower_bound_strike)]
    df_options = df_options[(df_options['strike'] <= upper_bound_strike)]
    df_options = df_options[(df_options['strike'] % option_step == 0)]
    df_options = df_options[(df_options['maturity'] <= maturity_filter )]
    print('After :', df_options.shape)

    # sparse option strikes and option types
    df_options_sparse = pd.concat([
            df_options[['strike', 'settlement', 'volume', 'open_interest', 'volatility', 'maturity']],
            pd.get_dummies(df_options['type']),
            # pd.get_dummies(df_options['strike'])
        ], axis=1)

    _max = 0
    for d in np.unique(df_index):
        # print(d)
        l = df_options[d:d].shape[0]
        # print(l)
        if (l > _max): _max = l
        # print(d, l, _max)
    print('>>>>', _max)

    df_options_sparse['price_distance_norm'] = abs(df_options_sparse.strike - df_index.close) / df_options_sparse.strike
    df_options_sparse['price_distance_sign'] = np.where(df_options_sparse.strike - df_index.close > 0, 1, 0)
    max_strike = df_options_sparse['strike'].max()
    max_settlement = df_options_sparse['settlement'].max()
    max_norm = np.max((max_settlement, max_strike))
    df_options_sparse['strike_norm'] = df_options_sparse['strike'] / max_norm
    df_options_sparse['settlement_norm'] = df_options_sparse['settlement'] / max_norm
    df_options_sparse['volume_norm'] = df_options_sparse['volume'] / df_options_sparse['volume'].max()
    df_options_sparse['open_interest_norm'] = df_options_sparse['open_interest'] / df_options_sparse[
        'open_interest'].max()
    
    
    opt_len = 0
    for date in np.unique(df_options_sparse.index.values):
        subset_shape = df_options_sparse[date:date].shape
        subset_len = subset_shape[0] * subset_shape[1]
        if (subset_len > opt_len):
            opt_len = subset_len

    option_labels = [
        ['o_' + str(l) + '_' + str(i) for l in df_options_sparse.columns.values]
        for i in range(1, int(opt_len / df_options_sparse.columns.shape[0]) + 1)]

    option_labels = np.array(option_labels, dtype=str)
    option_labels = np.reshape(option_labels, (option_labels.shape[0] * option_labels.shape[1]))

    option_data = None
    # recupero solo le date dove sono presenti quotazioni di borsa
    for date in np.unique(df_index.values):
        _val = np.zeros([1, opt_len])
        subset = df_options_sparse[date:date]
        values = np.reshape(subset.values, (1, subset.shape[0] * subset.shape[1]))
        _val[0, 0:values.shape[1]] = values
        if (option_data is None):
            option_data = _val
        else:
            option_data = np.vstack((option_data, _val))
    
    
    return pd.DataFrame(index=df_index, data=option_data, columns=option_labels);


def get_prices(conn, look_fwd):
    print(">>>> Process prices")
    sql_index = "select distinct strftime('%Y-%m-%d', ab.time/1000, 'unixepoch', 'localtime') as 'date', " \
                "ab.open, ab.high, ab.low, ab.close " \
                "from asset_bars ab " \
                "inner join future_bars fb on fb.time = ab.time"

    df_index = pd.read_sql(sql_index, con=conn, index_col=['date'])
    df_index.index = pd.to_datetime(df_index.index.values)
    df_index.sort_index(inplace=True)

    df_index_norm = df_index[['open', 'high', 'low', 'close']].pct_change(look_fwd)

    df_index_norm = pd.DataFrame(index=df_index.index[look_fwd:],
                                 data=df_index_norm,
                                 columns=['open', 'high', 'low', 'close'])

    df_index_norm['target'] = df_index_norm['close'].shift(-look_fwd)
    df_index_norm.dropna(axis=0, inplace=True)

    df_index_norm['dayofweek'] = df_index_norm.index.dayofweek + 1
    df_index_norm['month'] = df_index_norm.index.month
    df_index_norm['week'] = [weekofmonth(d) for d in df_index_norm.index.date]

    df_target_norm = df_index_norm['target']
    df_index_norm.drop(columns=['target'], axis=1, inplace=True)

    df_index_norm = pd.concat([
        df_index_norm[['open', 'high', 'low', 'close']],
        pd.get_dummies(df_index_norm['dayofweek']),
        pd.get_dummies(df_index_norm['week']),
        pd.get_dummies(df_index_norm['month'])
    ], axis=1)

    df_index_norm.columns.values[4:9] = ['mon', 'tue', 'wed', 'thu', 'fri']
    df_index_norm.columns.values[9:14] = ['week_1', 'week_2', 'week_3', 'week_4', 'week_5']
    df_index_norm.columns.values[14:26] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov','dec']
    
    return df_index_norm, df_target_norm

def main(conn, look_fwd=10, threshold=0.05, option_step=250.0):
    # GET PRINCES
    print(">>>> Process prices")
    sql_index = "select distinct strftime('%Y-%m-%d', ab.time/1000, 'unixepoch', 'localtime') as 'date', " \
                "ab.open, ab.high, ab.low, ab.close " \
                "from asset_bars ab " \
                "inner join future_bars fb on fb.time = ab.time"

    df_index = pd.read_sql(sql_index, con=conn, index_col=['date'])
    df_index.index = pd.to_datetime(df_index.index.values)
    df_index.sort_index(inplace=True)

    df_index_norm = df_index[['open', 'high', 'low', 'close']].pct_change(look_fwd)

    df_index_norm = pd.DataFrame(index=df_index.index[look_fwd:],
                                 data=df_index_norm,
                                 columns=['open', 'high', 'low', 'close'])

    df_index_norm['target'] = df_index_norm['close'].shift(-look_fwd)
    df_index_norm.dropna(axis=0, inplace=True)

    df_index_norm['dayofweek'] = df_index_norm.index.dayofweek + 1
    df_index_norm['month'] = df_index_norm.index.month
    df_index_norm['week'] = [weekofmonth(d) for d in df_index_norm.index.date]

    df_target_norm = df_index_norm['target']
    df_index_norm.drop(columns=['target'], axis=1, inplace=True)

    df_index_norm = pd.concat([
        df_index_norm[['open', 'high', 'low', 'close']],
        pd.get_dummies(df_index_norm['dayofweek']),
        pd.get_dummies(df_index_norm['week']),
        pd.get_dummies(df_index_norm['month'])
    ], axis=1)

    df_index_norm.columns.values[4:9] = ['mon', 'tue', 'wed', 'thu', 'fri']
    df_index_norm.columns.values[9:14] = ['week_1', 'week_2', 'week_3', 'week_4', 'week_5']
    df_index_norm.columns.values[14:26] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov',
                                           'dec']

    # GET OPTIONS
    print(">>>> Process options")
    sql_options = "select strftime('%Y-%m-%d', o.expire/1000, 'unixepoch', 'localtime') as 'expire', " \
                  "strftime('%Y-%m-%d', ob.time/1000, 'unixepoch', 'localtime') as 'date', " \
                  "o.type, o.strike, ob.volume, ob.open_interest, ob.settlement, " \
                  "ob.volatility/100 as 'volatility' " \
                  "from options o " \
                  "inner join option_bars ob on o.ticker = ob.ticker " \
                  "order by ob.time, o.expire, o.strike, o.type"

    df_options = pd.read_sql(sql_options, con=conn, index_col=['date'])
    df_options.index = pd.to_datetime(df_options.index.values)
    df_options[['volume', 'open_interest']] = df_options[['volume', 'open_interest']].astype(float)
    df_options.sort_index(inplace=True)
    df_options['expire'] = pd.to_datetime(df_options['expire'])

    df_options['maturity'] = (df_options['expire'] - df_options.index)
    df_options['maturity'] = (df_options['maturity'] / np.timedelta64(1, 'D')).astype(int) / 365.
    df_options.drop(columns=['expire'], axis=1, inplace=True)

    print('Before:', df_options.shape)
    df_options = df_options[(df_options['strike'] >= 10_000)]
    df_options = df_options[(df_options['strike'] <= 30_000)]
    df_options = df_options[(df_options['strike'] % option_step == 0)]
    df_options = df_options[(df_options['maturity'] <= 1.)]
    print('After :', df_options.shape)

    # sparse option strikes and option types
    df_options_sparse = pd.concat([
            df_options[['strike', 'settlement', 'volume', 'open_interest', 'volatility', 'maturity']],
            pd.get_dummies(df_options['type']),
            # pd.get_dummies(df_options['strike'])
        ], axis=1)

    _max = 0
    for d in np.unique(df_index_norm.index.values):
        # print(d)
        l = df_options[d:d].shape[0]
        # print(l)
        if (l > _max): _max = l
        # print(d, l, _max)
    print('>>>>', _max)

    df_options_sparse['price_distance_norm'] = abs(df_options_sparse.strike - df_index.close) / df_options_sparse.strike
    df_options_sparse['price_distance_sign'] = np.where(df_options_sparse.strike - df_index.close > 0, 1, 0)
    max_strike = df_options_sparse['strike'].max()
    max_settlement = df_options_sparse['settlement'].max()
    max_norm = np.max((max_settlement, max_strike))
    df_options_sparse['strike_norm'] = df_options_sparse['strike'] / max_norm
    df_options_sparse['settlement_norm'] = df_options_sparse['settlement'] / max_norm
    df_options_sparse['volume_norm'] = df_options_sparse['volume'] / df_options_sparse['volume'].max()
    df_options_sparse['open_interest_norm'] = df_options_sparse['open_interest'] / df_options_sparse[
        'open_interest'].max()


    # GET FUTURES
    print(">>>> Process futures")
    sql_future = "select strftime('%Y-%m-%d', f.expire/1000, 'unixepoch', 'localtime') as 'expire', " \
                 "strftime('%Y-%m-%d', fb.time/1000, 'unixepoch', 'localtime')  as 'date', " \
                 "fb.open_interest, fb.volume, fb.settlement " \
                 "from futures f " \
                 "inner join future_bars fb on f.ticker = fb.ticker " \
                 "order by fb.time, f.expire"

    df_future = pd.read_sql(sql_future, con=conn, index_col=['date'])
    df_future.index = pd.to_datetime(df_future.index.values)
    df_future.sort_index(inplace=True)
    # df_future['date'] = pd.to_datetime(df_future['date'])

    df_future['expire'] = pd.to_datetime(df_future['expire'])
    df_future[['open_interest', 'volume']] = df_future[['open_interest', 'volume']].astype(float)
    df_future['maturity'] = (df_future['expire'] - df_future.index.values)
    df_future['maturity'] = (df_future['maturity'] / np.timedelta64(1, 'D')).astype(int) / 365.

    # Future scalers
    futureOpenInterestScaler = MinMaxScaler()
    futureVolumeScaler = MinMaxScaler()
    # Index and future's settlement scaler
    indexPriceScaler = MinMaxScaler()

    df_future['open_interest'] = futureOpenInterestScaler.fit_transform(np.reshape(
        df_future['open_interest'].values, (df_future['open_interest'].values.shape[0], 1)))
    df_future['volume'] = futureVolumeScaler.fit_transform(
        np.reshape(df_future['volume'].values, (df_future['volume'].values.shape[0], 1)))
    df_future['settlement'] = indexPriceScaler.fit_transform(
        np.reshape(df_future['settlement'].values, (df_future['settlement'].values.shape[0], 1)))
    df_future.drop(columns=['expire'], axis=1, inplace=True)


    future_data = None
    for date in np.unique(df_index_norm.index.values):
        _val = np.zeros([1, 16])
        subset = df_future[date:date]
        values = np.reshape(subset.values, (1, subset.shape[0] * subset.shape[1]))
        _val[0, 0:values.shape[1]] = values
        if (future_data is None):
            future_data = _val
        else:
            future_data = np.vstack((future_data, _val))

    future_labels = np.array([['f_' + l + '_' + str(i) for l in df_future.columns.values] for i in range(1, 5)],
                             dtype=str)
    future_labels = np.reshape(future_labels, (future_labels.shape[0] * future_labels.shape[1]))

    opt_len = 0
    for date in np.unique(df_options_sparse.index.values):
        subset_shape = df_options_sparse[date:date].shape
        subset_len = subset_shape[0] * subset_shape[1]
        if (subset_len > opt_len):
            opt_len = subset_len

    option_labels = [
        ['o_' + str(l) + '_' + str(i) for l in df_options_sparse.columns.values]
        for i in range(1, int(opt_len / df_options_sparse.columns.shape[0]) + 1)]

    option_labels = np.array(option_labels, dtype=str)
    option_labels = np.reshape(option_labels, (option_labels.shape[0] * option_labels.shape[1]))

    option_data = None
    # recupero solo le date dove sono presenti quotazioni di borsa
    for date in np.unique(df_index_norm.index.values):
        _val = np.zeros([1, opt_len])
        subset = df_options_sparse[date:date]
        values = np.reshape(subset.values, (1, subset.shape[0] * subset.shape[1]))
        _val[0, 0:values.shape[1]] = values
        if (option_data is None):
            option_data = _val
        else:
            option_data = np.vstack((option_data, _val))

    index_data = df_index_norm.values
    target_data = df_target_norm.values

    x_data = np.hstack((index_data, future_data, option_data))
    y_data = target_data  # np.reshape(target_data, (target_data.shape[0], 1))
    y_data_sparse = np.where(y_data > threshold, 1, np.where(y_data < -threshold, -1., 0.))

    index_labels = df_index_norm.columns.values.astype(str)
    x_labels = np.hstack((index_labels, future_labels, option_labels))
    #x_labels = option_labels

    print("X Dataset shape", x_data.shape)
    print("Y Dataset shape", y_data.shape)

    data = np.hstack((
        x_data,
        np.reshape(y_data, (-1, 1)),
        np.reshape(y_data_sparse, (-1, 1))))

    labels = np.hstack((x_labels, ('y_data', 'y_data_sparse')))
    data_index = df_index_norm.index

    df = pd.DataFrame(index=data_index, data=data, columns=labels)
    df.reset_index().to_csv('./trasi.csv')

    return df


if __name__ == '__main__':
    conn = sqlite3.connect('../data/trasi.db')
    main(conn)
