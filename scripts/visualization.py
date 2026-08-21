import numpy as np
import matplotlib.pyplot as plt


def plot_reliability_diagram(
    y_true,
    y_prob,
    model_label,
    title,
    n_bins=10
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Define probability bins
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Assign each probability to a bin
    bin_ids = np.digitize(
        y_prob,
        bin_edges[1:-1]
    )

    mean_predicted_prob = []
    fraction_of_positives = []

    # Compute calibration values for each bin
    for i in range(n_bins):

        mask = bin_ids == i

        if np.sum(mask) > 0:

            mean_predicted_prob.append(
                np.mean(y_prob[mask])
            )

            fraction_of_positives.append(
                np.mean(y_true[mask])
            )

    # Plot
    plt.figure(figsize=(6, 6))

    # Perfect calibration line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    # Model reliability curve
    plt.plot(
        mean_predicted_prob,
        fraction_of_positives,
        marker="o",
        label=model_label
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()



def compare_reliability_diagrams(
    y_true,
    y_prob_before,
    y_prob_after,
    label_before="Before calibration",
    label_after="After calibration",
    title="Reliability Diagram Comparison",
    n_bins=10
):
    y_true = np.asarray(y_true)
    y_prob_before = np.asarray(y_prob_before)
    y_prob_after = np.asarray(y_prob_after)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    def compute_reliability(y_prob):
        bin_ids = np.digitize(
            y_prob,
            bin_edges[1:-1]
        )

        mean_predicted_prob = []
        fraction_of_positives = []

        for i in range(n_bins):
            mask = bin_ids == i

            if np.sum(mask) > 0:
                mean_predicted_prob.append(
                    np.mean(y_prob[mask])
                )

                fraction_of_positives.append(
                    np.mean(y_true[mask])
                )

        return mean_predicted_prob, fraction_of_positives

    # Before calibration
    mean_before, frac_before = compute_reliability(
        y_prob_before
    )

    # After calibration
    mean_after, frac_after = compute_reliability(
        y_prob_after
    )

    # Plot
    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        mean_before,
        frac_before,
        marker="o",
        label=label_before
    )

    plt.plot(
        mean_after,
        frac_after,
        marker="o",
        label=label_after
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


def compare_all_calibrations(
    y_true,
    y_prob_before,
    y_prob_platt,
    y_prob_isotonic,
    title="Calibration Comparison",
    n_bins=5
):
    y_true = np.asarray(y_true)
    y_prob_before = np.asarray(y_prob_before)
    y_prob_platt = np.asarray(y_prob_platt)
    y_prob_isotonic = np.asarray(y_prob_isotonic)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Compute the reliability curve for a set of probabilities
    def compute_reliability(y_prob):

        bin_ids = np.digitize(
            y_prob,
            bin_edges[1:-1]
        )

        mean_predicted_prob = []
        fraction_of_positives = []

        for i in range(n_bins):

            mask = bin_ids == i

            if np.sum(mask) > 0:

                mean_predicted_prob.append(
                    np.mean(y_prob[mask])
                )

                fraction_of_positives.append(
                    np.mean(y_true[mask])
                )

        return (
            mean_predicted_prob,
            fraction_of_positives
        )

    mean_before, frac_before = compute_reliability(
        y_prob_before
    )
    mean_platt, frac_platt = compute_reliability(
        y_prob_platt
    )
    mean_isotonic, frac_isotonic = compute_reliability(
        y_prob_isotonic
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(mean_before, frac_before, marker="o", label="Before calibration")
    plt.plot(mean_platt, frac_platt, marker="o", label="Platt Scaling")
    plt.plot(mean_isotonic, frac_isotonic, marker="o", label="Isotonic Regression")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")

    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
