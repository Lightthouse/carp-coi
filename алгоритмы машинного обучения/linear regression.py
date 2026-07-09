import pandas as pd
import numpy as np
import random


class MyLineReg:
    def __init__(self, n_iter: int = 100, learning_rate=0.1, weights: list = None, metric: str = None,
                 reg: str = None, l1_coef: float = 0.0, l2_coef: float = 0.0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.loss = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def compute_loss(self, y, y_pred, weights):
        metrics_dict = {
            'mse': lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'mae': lambda y, y_pred: np.mean(abs(y - y_pred)),
            'rmse': lambda y, y_pred: np.mean((y - y_pred) ** 2) ** 0.5,
            'r2': lambda y, y_pred: 1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2),
            'mape': lambda y, y_pred: np.mean(abs((y - y_pred) / y)) * 100,
        }

        loss_function = metrics_dict.get(self.metric, metrics_dict['mse'])
        loss = loss_function(y, y_pred)

        if self.reg == 'l1':
            return loss + self.l1_coef * sum(abs(weights))
        if self.reg == 'l2':
            return loss + self.l2_coef * sum(weights ** 2)
        if self.reg == 'elasticnet':
            return loss + self.l1_coef * sum(abs(weights)) + self.l2_coef * sum(weights ** 2)
        return loss

    def compute_grad(self, grad, weights):
        if self.reg == 'l1':
            return grad + self.l1_coef * np.sign(weights)
        if self.reg == 'l2':
            return grad + self.l2_coef * 2 * weights
        if self.reg == 'elasticnet':
            return grad + self.l1_coef * (np.sign(weights)) + self.l2_coef * 2 * weights
        return grad

    def _get_learning_rate(self, iteration):
        """Получение learning rate на текущей итерации"""
        if callable(self.learning_rate):
            return self.learning_rate(iteration)
        else:
            return self.learning_rate

    def _get_sample_size(self, n_samples):
        """Определение размера мини-пакета"""
        if self.sgd_sample is None:
            return n_samples  # полный датасет

        if isinstance(self.sgd_sample, int):
            return self.sgd_sample
        else:  # дробное число
            return int(n_samples * self.sgd_sample)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        # Фиксируем сид для воспроизводимости
        random.seed(self.random_state)

        # Копия + единичный столбец слева
        X = X.copy()
        X.insert(0, 'x_once', 1.0)

        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)

        # Сохраняем полные данные
        X_full = X.values
        y_full = y.values

        # Определяем размер мини-пакета
        batch_size = self._get_sample_size(n_samples)

        for i in range(1, self.n_iter + 1):
            # Формируем мини-пакет
            if batch_size < n_samples:
                # Случайная выборка индексов
                sample_rows_idx = random.sample(range(n_samples), batch_size)
                X_batch = X_full[sample_rows_idx]
                y_batch = y_full[sample_rows_idx]
            else:
                # Используем все данные
                X_batch = X_full
                y_batch = y_full

            # Предсказание на мини-пакете
            y_pred = X_batch @ self.weights
            error = y_pred - y_batch

            # Вычисляем loss на ПОЛНОМ датасете (для логирования)
            y_pred_full = X_full @ self.weights
            loss = self.compute_loss(y_full, y_pred_full, weights=self.weights)

            if verbose and i == 1:
                print(f"Initial loss: {loss}")

            # Вычисляем градиент на мини-пакете
            grad = (2 / batch_size) * (X_batch.T @ error)
            grad = self.compute_grad(grad, self.weights)

            # Обновляем веса
            lr = self._get_learning_rate(i)
            self.weights = self.weights - lr * grad

            # Вычисляем новый loss на ПОЛНОМ датасете
            y_pred_full_new = X_full @ self.weights
            loss_new = self.compute_loss(y_full, y_pred_full_new, weights=self.weights)

            if verbose and i % verbose == 0:
                print(f"Iteration {i}: loss = {loss_new}")

            self.loss = loss_new

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def get_best_score(self):
        return self.loss

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'x_once', 1.0)
        return X.values @ self.weights  # возвращаем массив, а не сумму

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'