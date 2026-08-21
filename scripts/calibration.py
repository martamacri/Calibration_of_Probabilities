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

    def fit(self, scores, y): # important part --> learn the calibration
        scores = np.asarray(scores, dtype=float) # NumPy
        y = np.asarray(y)

        targets = self._smooth_targets(y) # with new value

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

    def predict_proba(self, scores): # application on test set
        scores = np.asarray(scores, dtype=float)

        linear_output = self.A * scores + self.B

        return self._sigmoid(linear_output)



class IsotonicRegression:
    def __init__(self):
        self.thresholds = None # score
        self.values = None # associated probabilities

    def fit(self, scores, y): # learn the calibration
        scores = np.asarray(scores, dtype=float) # NumPy
        y = np.asarray(y, dtype=float)

        order = np.argsort(scores)

        scores_sorted = scores[order]
        y_sorted = y[order]

        blocks = []

        for score, target in zip(scores_sorted, y_sorted):
            blocks.append({
                "min_score": score,
                "max_score": score,
                "sum_y": target,
                "weight": 1
            }) # for every osservation 

        i = 0

        while i < len(blocks) - 1: # comparison with next block for the monotonicity of score and probability

            current_value = (blocks[i]["sum_y"] / blocks[i]["weight"])
            next_value = (blocks[i + 1]["sum_y"] / blocks[i + 1]["weight"])

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

        self.thresholds = []
        self.values = []

        for block in blocks:

            self.thresholds.append(block["max_score"])
            self.values.append(block["sum_y"] / block["weight"])

        self.thresholds = np.asarray(self.thresholds)
        self.values = np.asarray(self.values) # NumPy

        return self # trained calibrator

    def predict_proba(self, scores): # application on test set

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