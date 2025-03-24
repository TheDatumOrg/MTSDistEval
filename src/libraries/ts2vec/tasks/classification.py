import numpy as np
from . import _eval_protocols as eval_protocols

def eval_classification(model, X_train, y_train, X_test, y_test):
    assert y_train.ndim == 1 or y_train.ndim == 2
    train_repr = model.encode(X_train, encoding_window='full_series' if y_train.ndim == 1 else None)
    test_repr = model.encode(X_test, encoding_window='full_series' if y_train.ndim == 1 else None)

    fit_clf = eval_protocols.fit_knn

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if y_train.ndim == 2:
        train_repr = merge_dim01(train_repr)
        y_train = merge_dim01(y_train)
        test_repr = merge_dim01(test_repr)
        y_test = merge_dim01(y_test)

    clf = fit_clf(train_repr, y_train)
    acc = clf.score(test_repr, y_test)
    return acc
