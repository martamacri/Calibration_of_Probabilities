"""Visualization utilities for calibration analysis."""

import numpy as np
import matplotlib.pyplot as plt


def _compute_reliability(y_true, y_prob, n_bins):
    """Compute mean predicted probabilities and observed positive fractions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)

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


def plot_reliability_diagram(
    y_true,
    y_prob,
    model_label,
    title,
    n_bins=10
):
    """Plot a reliability diagram for predicted probabilities."""

    mean_predicted_prob, fraction_of_positives = _compute_reliability(
        y_true,
        y_prob,
        n_bins
    )

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
    """Compare reliability diagrams before and after calibration."""

    mean_before, frac_before = _compute_reliability(
        y_true,
        y_prob_before,
        n_bins
    )

    mean_after, frac_after = _compute_reliability(
        y_true,
        y_prob_after,
        n_bins
    )

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
    """Compare uncalibrated, Platt-scaled, and isotonic probabilities."""

    mean_before, frac_before = _compute_reliability(
        y_true,
        y_prob_before,
        n_bins
    )

    mean_platt, frac_platt = _compute_reliability(
        y_true,
        y_prob_platt,
        n_bins
    )

    mean_isotonic, frac_isotonic = _compute_reliability(
        y_true,
        y_prob_isotonic,
        n_bins
    )

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
        label="Before calibration"
    )

    plt.plot(
        mean_platt,
        frac_platt,
        marker="o",
        label="Platt Scaling"
    )

    plt.plot(
        mean_isotonic,
        frac_isotonic,
        marker="o",
        label="Isotonic Regression"
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
