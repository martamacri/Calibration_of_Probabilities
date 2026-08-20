import numpy as np

# Platt Scaling

class PlattScaling:
    def __init__(self, learning_rate=0.01, max_iter=10000, tolerance=1e-7):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.A = 0.0
        self.B = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _smooth_targets(y): # no 0,1 but other value --> no overfitting
        y = np.asarray(y)

        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)

        positive_target = (n_positive + 1) / (n_positive + 2) # value for no overfitting
        negative_target = 1 / (n_negative + 2)

        return np.where(
            y == 1,
            positive_target,
            negative_target
        ) # new target value

    def fit(self, scores, y): # important part
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)

        targets = self._smooth_targets(y)

        self.A = 0.0
        self.B = 0.0

        n = len(scores)

        for _ in range(self.max_iter):

            linear_output = self.A * scores + self.B
            probabilities = self._sigmoid(linear_output)

            error = probabilities - targets

            gradient_A = np.sum(error * scores) / n
            gradient_B = np.sum(error) / n

            new_A = self.A - self.learning_rate * gradient_A
            new_B = self.B - self.learning_rate * gradient_B

            if (
                abs(new_A - self.A) < self.tolerance
                and abs(new_B - self.B) < self.tolerance
            ):
                self.A = new_A
                self.B = new_B
                break

            self.A = new_A
            self.B = new_B

        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float)

        linear_output = self.A * scores + self.B

        return self._sigmoid(linear_output)