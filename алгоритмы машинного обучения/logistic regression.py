import random

import pandas as pd
import numpy as np


class MyLogReg:
    def __init__(self, n_iter: int = 10,
                 learning_rate: float = 0.1,
                 weights: np.ndarray = None,
                 metric: str = None,
                 reg: str = None,
                 l1_coef: float = 0.0,
                 l2_coef: float = 0.0,
                 sgd_sample=None,
                 random_state=42
                 ):

        self.n_iter = n_iter
        self.lr = learning_rate
        self.weights = weights
        self.metric_name = metric
        self.metric = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

        self.eps = 1e-15  # чтобы не было логарифма нуля


    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.lr}'

    def _proba_to_class(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred_proba = y_pred > 0.5
        return y_pred_proba.astype(int)

    def _accuracy(self, tp: float, fp: float, tn: float, fn: float) -> float:
        return (tp + tn) / (tp + tn + fp + fn)

    def _f1_score(self, tp: float, fp: float, fn: float) -> float:
        precision = self._precision(tp, fp)
        recall = self._recall(tp, fn)

        if precision + recall == 0:
            return 0

        return 2 * (precision * recall) / (precision + recall)

    def _precision(self, tp: float, fp: float) -> float:
        return tp / (tp + fp)

    def _recall(self, tp: float, fn: float) -> float:
        return tp / (tp + fn)

    def _roc_auc(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        total_pos = np.sum(y == 1)
        total_neg = np.sum(y == 0)

        if total_pos == 0 or total_neg == 0:
            return 0.0

        y_pred = np.round(y_pred, 10)
        sorted_indices = np.argsort(y_pred)[::-1]
        y_sorted = y[sorted_indices]
        score_sorted = y_pred[sorted_indices]

        roc_auc = 0.0
        pos_seen = 0
        i = 0
        n = len(y_sorted)

        while i < n:
            j = i
            while j < n and score_sorted[j] == score_sorted[i]:
                j += 1

            group_positive = np.sum(y_sorted[i:j])
            group_negative = (j - i) - group_positive

            roc_auc += group_negative * pos_seen + group_negative * group_positive / 2
            pos_seen += group_positive
            i = j

        return roc_auc / (total_pos * total_neg)

    def _compute_metric(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        metric = 0.0
        if not self.metric_name:
            return metric

        if self.metric_name == 'roc_auc':
            metric = self._roc_auc(y, y_pred)
            return metric

        y_pred_class = self._proba_to_class(y_pred)

        tp = np.sum((y_pred_class + y) == 2)
        fp = np.sum(y_pred_class > y)
        tn = np.sum((y_pred_class + y) == 0)
        fn = np.sum(y_pred_class < y)

        if self.metric_name == 'accuracy':
            metric = self._accuracy(tp, fp, tn, fn)

        if self.metric_name == 'f1':
            metric = self._f1_score(tp, fp, fn)

        if self.metric_name == 'precision':
            metric = self._precision(tp, fp)

        if self.metric_name == 'recall':
            metric = self._recall(tp, fn)

        return metric

    def _make_prediction(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X @ weights))

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        eps = self.eps
        return -1 * np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

    def _verbose(self, verbose, iter_count, X, y, weights):
        if verbose and iter_count % verbose == 0:
            y_pred = self._make_prediction(X, weights)
            log_loss = self._compute_loss(y, y_pred)
            self.metric = self._compute_metric(y, y_pred)

            metric_info = f' | {self.metric_name}: {self.metric}' if self.metric_name else ''
            print(f'iter: {iter_count}, loss: {log_loss}{metric_info}')

    def _compute_regularization(self, weights) -> float:
        if (self.reg is None) or (self.reg not in ['l1', 'l2', 'elasticnet']):
            return 0.0

        if self.reg == 'l1':
            return self.l1_coef * np.sign(weights)

        if self.reg == 'l2':
            return self.l2_coef * 2 * weights

        if self.reg == 'elasticnet':
            return self.l1_coef * np.sign(weights) + self.l2_coef * 2 * weights

        return 0.0

    def get_best_score(self):
        return self.metric

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):

        X = X.copy()
        X.insert(0, 'x_bias', 1)

        X = X.values
        y = y.values

        n_samples, n_features = X.shape
        weights = self.weights if self.weights is not None else np.ones(n_features)

        random.seed(self.random_state)
        sgd_sample = int(n_samples * self.sgd_sample) if isinstance(self.sgd_sample, float) else self.sgd_sample

        batch_size = n_samples if not sgd_sample else sgd_sample

        iter_count = 0

        for iter_ in range(self.n_iter):
            iter_count += 1

            sample_rows_idx = random.sample(range(X.shape[0]), batch_size)
            X_batch = X[sample_rows_idx] if sgd_sample else X
            y_batch = y[sample_rows_idx] if sgd_sample else y

            y_pred_batch = self._make_prediction(X_batch, weights)

            y_pred = self._make_prediction(X, weights)
            self.metric = self._compute_metric(y, y_pred)
            # log_loss = self._compute_loss(y, y_pred)  в этом задании не используем

            regularization = self._compute_regularization(weights)
            grad = (1 / batch_size) * (X_batch.T @ (y_pred_batch - y_batch)) + regularization

            lr = self.lr(iter_count) if callable(self.lr) else self.lr
            weights = weights - lr * grad

            # отладка
            self._verbose(verbose, iter_count, X, y, weights)

        # обновим веса и метрики исходя из последних подсчетов
        y_pred = self._make_prediction(X, weights)
        self._compute_metric(y, y_pred)
        self.weights = weights

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def predict_proba(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'x_once', 1.0)
        X = X.values
        y_pred = 1 / (1 + np.exp(-X @ self.weights))
        return y_pred

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'x_once', 1.0)
        X = X.values
        y_pred = 1 / (1 + np.exp(-X @ self.weights))
        return self._proba_to_class(y_pred)
