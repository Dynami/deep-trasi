import model_selection.model_selection as ms
import numpy as np


def scoring(model, x_test, y_test):
    y_pred = model.predict(x_test)

    predict_proba = getattr(model, "predict_proba", None)

    if callable(predict_proba):
        y_pred_proba = model.predict_proba(x_test)
        y_pred_proba = [y_pred_proba[i, v] for i, v in enumerate(np.argmax(y_pred_proba, axis=1))]
        y_pred_proba = np.array(y_pred_proba)
    else:
        y_pred_proba = np.ones((x_test.shape[0]))

    y_test_data_sparse = y_test[['sparse_target']].values
    y_test_data_sparse = np.reshape(y_test_data_sparse, (-1))
    y_test_data_return = y_test[['target']].values
    y_test_data_return = np.reshape(y_test_data_return, (-1))

    # print(y_pred.shape, y_test_data_sparse.shape, y_pred_proba.shape, y_test_data_return.shape)
    result = y_pred * y_pred_proba * y_test_data_return
    # non zero measurament
    nz = result[np.where(result != 0)]
    nz_coverage = (float(len(nz)) / len(result))
    nz_mean = np.mean(nz)
    nz_std = np.std(nz)
    sharpe_ratio = nz_mean / nz_std
    # Precision score
    nz_prec = np.sum(np.where(nz > 0, 1, 0)) / float(len(nz))

    score = sharpe_ratio*nz_coverage

    print('Detail>>>>>', result)
    print('Total >>>>> %.3f%%' % (np.sum(result) * 100))
    print('Prec. >>>>> %.3f%%' % (nz_prec * 100))
    print('Cover.>>>>> %.3f%%' % (nz_coverage * 100))
    print('Sharp >>>>> %.3f' % sharpe_ratio)
    print('Score >>>>> %.3f' % score)
    print('Mean  >>>>> %.3f' % nz_mean)
    print('Std   >>>>> %.3f' % nz_std)
    print('Samples>>>> %s' % x_test.shape[0])

    scores = {
        'total_return': np.sum(result),
        'precision': nz_prec,
        'coverage': nz_coverage,
        'sharpe_ratio': sharpe_ratio,
        'score': score,
        'mean': nz_mean,
        'std': nz_std
    }
    return scores, y_pred


def main(model, x_test, y_test, look_fwd, investment=1000.0, take_profit=0.05, stop_loss=-0.02):
    single_bet = investment/look_fwd
    score, y_pred = scoring(model['model'], x_test, y_test)
    print(score)
    y_true = x_test['close_1']
    y_target = y_test['target']

    pass