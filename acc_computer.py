import numpy as np

def _fast_hist(targets, preds, n_class):
    mask = (targets >= 0) & (targets < n_class)
    hist = np.bincount(n_class * targets[mask].astype(int) +
                       preds[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy(targets, preds, n_class):
    hist = np.zeros((n_class, n_class))
    for tar_i, pred_i in zip(targets, preds):
        hist += _fast_hist(tar_i.flatten(), pred_i.flatten(), n_class)

    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc