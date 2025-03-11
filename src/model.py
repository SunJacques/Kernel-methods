import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from src.tools import Score

class KSVM():
    def __init__(self, kernel, C, tol=1e-5, name="KSVM"):
        
        self.kernel = kernel()
        self.__name__ = name

        self.C = C
        self.tol = tol

    def predict(self, data):
        K_xi = self.kernel.predict(data)
        pred = self.alpha @ K_xi.T + self.b
        return np.sign(pred)

    def score_recall_precision(self, data, labels):
        mask = np.arange(len(data))
        np.random.shuffle(mask)
        pred = self.predict(data[mask])
        score = Score(pred, labels[mask])
        return score
    
    def fit(self, X, y):
        """
        Train the Support Vector Machine (SVM) model using a kernel method.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            tuple: (alphas, bias) where:
                - alphas (np.ndarray): Lagrange multipliers.
                - bias (float): Intercept term.
        """
        self.kernel.fit(X)
        K = self.kernel.get_kernel()
        n_samples = len(K)
        C = self.C

        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]), (2 * n_samples, n_samples), 'd')
        h = matrix(np.hstack([np.zeros(n_samples), np.ones(n_samples) * C]), (2 * n_samples, 1), 'd')
        P = matrix(np.dot(np.diag(y), np.dot(K, np.diag(y))), (n_samples, n_samples), 'd')
        q = matrix(-np.ones(n_samples), (n_samples, 1), 'd')

        A = matrix(y, (1, n_samples), "d")
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        alpha = y * np.array(solution['x']).reshape(-1)
        support_vectors = np.where(np.abs(alpha) > self.tol)[0]

        intercept = 0
        for sv in support_vectors:
            intercept += y[sv]
            intercept -= np.sum(
                alpha[support_vectors, None] * K[sv, support_vectors])
        if len(support_vectors) > 0:
            intercept /= len(support_vectors)

        # set to zero non support vectors
        alpha[np.where(np.abs(alpha) <= self.tol)[0]] = 0
        
        self.b = intercept
        self.alpha = alpha
        return self.alpha, self.b
