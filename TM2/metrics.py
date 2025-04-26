# metrics.py（保持不变）
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def classification_metrics(pred, true, threshold=0.5, target_dim=-1):
    if pred.ndim > 1:
        pred = pred[..., target_dim]
    if true.ndim > 1:
        true = true[..., target_dim]

    pred = pred.flatten()
    true = true.flatten()

    pred_labels = (pred > threshold).astype(int)
    true_labels = (true > threshold).astype(int)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    return accuracy, precision, f1


def metric(pred, true, threshold=None, target_dim=-1):
    metrics_dict = {
        'mae': MAE(pred, true),
        'mse': MSE(pred, true),
        'rmse': RMSE(pred, true),
        'mape': MAPE(pred, true),
        'mspe': MSPE(pred, true)
    }

    if threshold is not None:
        acc, pre, f1 = classification_metrics(pred, true, threshold, target_dim)
        metrics_dict.update({
            'accuracy': acc,
            'precision': pre,
            'f1_score': f1
        })

    return metrics_dict