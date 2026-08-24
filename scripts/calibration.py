"""Calibration methods implemented from scratch."""
import numpy as np

# Platt Scaling

class PlattScaling:
    """Platt Scaling calibrator implemented from scratch."""
    def __init__(self, learning_rate=0.01, max_iter=10000, tolerance=1e-7):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.a = 0.0
        self.b = 0.0

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

    def fit(self, scores, y): # important part --> learn the calibration
        """Fit the calibration mapping on calibration data."""
        scores = np.asarray(scores, dtype=float) # NumPy
        y = np.asarray(y)

        targets = self._smooth_targets(y) # with new value

        self.a = 0.0
        self.b = 0.0

        n = len(scores)

        for _ in range(self.max_iter):

            linear_output = self.a * scores + self.b
            probabilities = self._sigmoid(linear_output)

            error = probabilities - targets

            gradient_a = np.sum(error * scores) / n
            gradient_b = np.sum(error) / n

            new_a = self.a - self.learning_rate * gradient_a
            new_b = self.b - self.learning_rate * gradient_b

            if (
                abs(new_a - self.a) < self.tolerance
                and abs(new_b - self.b) < self.tolerance
            ):
                self.a = new_a
                self.b = new_b
                break

            self.a = new_a
            self.b = new_b

        return self

    def predict_proba(self, scores): # application on test set
        """Return calibrated probabilities."""
        scores = np.asarray(scores, dtype=float)

        linear_output = self.a * scores + self.b

        return self._sigmoid(linear_output)



class IsotonicRegression:
    """Isotonic Regression calibrator implemented from scratch."""
    def __init__(self):
        self.thresholds = None # score
        self.values = None # associated probabilities

    def fit(self, scores, y): # learn the calibration
        """Fit the calibration mapping on calibration data."""
        scores = np.asarray(scores, dtype=float) # NumPy
        y = np.asarray(y, dtype=float)

        order = np.argsort(scores)

        scores_sorted = scores[order]
        y_sorted = y[order]

        unique_scores, inverse = np.unique( # Group observations with identical scores
            scores_sorted,
            return_inverse=True
        )

        sum_y = np.zeros(len(unique_scores))
        weights = np.zeros(len(unique_scores))

        for i, group_index in enumerate(inverse):
            sum_y[group_index] += y_sorted[i]
            weights[group_index] += 1

        blocks = []

        for score, target_sum, weight in zip(
            unique_scores,
            sum_y,
            weights
        ):
            blocks.append({
                "min_score": score,
                "max_score": score,
                "sum_y": target_sum,
                "weight": weight
            }) # for every osservation

        i = 0

        while i < len(blocks) - 1: # comparison with next block for the monotonicity

            current_value = blocks[i]["sum_y"] / blocks[i]["weight"]
            next_value = blocks[i + 1]["sum_y"] / blocks[i + 1]["weight"]

            if current_value > next_value: # merge blocks if it not greater

                merged_block = {
                    "min_score": blocks[i]["min_score"],
                    "max_score": blocks[i + 1]["max_score"],
                    "sum_y": blocks[i]["sum_y"] + blocks[i + 1]["sum_y"],
                    "weight":blocks[i]["weight"] + blocks[i + 1]["weight"]
                }

                blocks[i] = merged_block
                del blocks[i + 1]

                if i > 0:
                    i -= 1

            else:
                i += 1

        self.thresholds = np.asarray([
            block["max_score"]
            for block in blocks
        ])

        self.values = np.asarray([
            block["sum_y"] / block["weight"]
            for block in blocks
        ])

        return self # trained calibrator

    def predict_proba(self, scores): # application on test set
        """Return calibrated probabilities."""
        scores = np.asarray(scores, dtype=float)
        probabilities = np.empty(len(scores))

        for i, score in enumerate(scores):

            index = np.searchsorted(
                self.thresholds,
                score,
                side="left"
            )

            if index >= len(self.values):
                index = len(self.values) - 1

            probabilities[i] = self.values[index]

        return probabilities
