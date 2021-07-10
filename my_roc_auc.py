#modified from https://github.com/SiLiKhon/my_roc_auc/blob/master/my_roc_auc.py
from __future__ import division
from __future__ import absolute_import
import numpy as np
from itertools import izip

def my_roc_auc(classes,
               predictions,
               weights = None):
    u"""
    Calculating ROC AUC score as the probability of correct ordering
    """

    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
            shape=len(classes),
            dtype=[(u'c', classes.dtype),
                   (u'p', predictions.dtype),
                   (u'w', weights.dtype)]
        )
    data[u'c'], data[u'p'], data[u'w'] = classes, predictions, weights

    data = data[np.argsort(data[u'c'])]
    data = data[np.argsort(data[u'p'], kind=u'mergesort')] # here we're relying on stability as we need class orders preserved

    correction = 0.
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data[u'p'][1:] == data[u'p'][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        ids, = mask2.nonzero()
        correction = sum([((dsplit[u'c'] == class0) * dsplit[u'w'] * msplit).sum() * 
                          ((dsplit[u'c'] == class1) * dsplit[u'w'] * msplit).sum()
                          for dsplit, msplit in izip(np.split(data, ids), np.split(mask1, ids))]) * 0.5
 
    weights_0 = data[u'w'] * (data[u'c'] == class0)
    weights_1 = data[u'w'] * (data[u'c'] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (weights_1.sum() * cumsum_0[-1])

