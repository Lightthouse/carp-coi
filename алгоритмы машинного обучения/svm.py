import numpy as np
import pandas as pd
import random


class MySVM:
    def __init__(self, n_iter = 10, learning_rate = 0.001, C = 1, sgd_sample = None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = 1
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return 'MySVM class: n_iter={}, learning_rate={}'.format(self.n_iter, self.learning_rate)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False):
        random.seed(self.random_state)

        X = X.copy()
        X = X.values

        y = y.values
        y = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape
        weights = np.ones(n_features)
        b = 1

        sgd_sample = int(n_samples * self.sgd_sample) if isinstance(self.sgd_sample, float) else self.sgd_sample
        batch_size = n_samples if not sgd_sample else sgd_sample

        for n in range(1, self.n_iter + 1):
            sample_rows_idx = random.sample(range(n_samples), batch_size)
            X_batch = X if not sgd_sample else X[sample_rows_idx]
            y_batch = y if not sgd_sample else y[sample_rows_idx]

            for i in range(n_samples):
                xi = X_batch[i]
                yi = y_batch[i]

                correct_sign = (yi * (weights @ xi + b)) >= 1
                grad_w = 2 * weights if correct_sign else 2 * weights - self.C * yi * xi
                grad_b = 0 if correct_sign else -self.C * yi

                weights -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b

            if verbose and (n % verbose == 0 or n == 1):
                loss = weights @ weights + self.C * np.mean(np.maximum(0, 1- y * (weights @ X + b)))
                print(f'{n}| loss: {loss}')

        self.weights = weights
        self.b = b

    def get_coef(self):
        return self.weights, self.b

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X = X.values
        y = np.sign(X @ self.weights + self.b)
        y = np.where(y == -1, 0, 1)
        return y