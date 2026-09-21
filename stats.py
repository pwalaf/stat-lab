"""Logique mathématique et rendu graphique pour StatLab."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, norm

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.grid": False,
})


def compute_binomial_approximation(n: int, p: float):
    """Calcule les paramètres, lois et figures pour De Moivre-Laplace."""
    mean = n * p
    variance = n * p * (1 - p)
    sigma = np.sqrt(variance)
    is_valid = (mean > 5) and (n * (1 - p) > 5)

    k_values = np.arange(0, n + 1)
    lo = max(0, mean - 4 * sigma)
    hi = min(n, mean + 4 * sigma)
    x_curve = np.linspace(lo, hi, 400)
    y_curve = norm.pdf(x_curve, mean, sigma)

    return {
        "mean": mean,
        "variance": variance,
        "is_valid": is_valid,
        "k_values": k_values,
        "x_curve": x_curve,
        "y_curve": y_curve,
        "bounds": (lo, hi),
    }


def plot_binomial_distribution(k_values, bar_values, x_curve, y_curve, bounds):
    """Génère la figure de la loi binomiale vs normale."""
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(k_values, bar_values, width=0.8, color="#3f6b4a", alpha=0.35, label="Distribution")
    ax.plot(x_curve, y_curve, color="#3f6b4a", linewidth=1.6, label="Approximation normale")
    ax.set_xlim(bounds)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    plt.close(fig)
    return fig


def prepare_imported_sim_table(last_sim: dict) -> pd.DataFrame:
    """Transforme les données de simulation Monte-Carlo en classes pour le Chi2."""
    n_s, p_s = last_sim["n"], last_sim["p"]
    counts, total = last_sim["counts"], last_sim["total"]
    pmf = binom.pmf(np.arange(0, n_s + 1), n_s, p_s)

    classes = []
    i = 0
    while i <= n_s:
        acc, acc_p, start = counts[i], pmf[i], i
        while acc_p * total < 5 and i < n_s:
            i += 1
            acc += counts[i]
            acc_p += pmf[i]
        label = f"X = {start}" if i == start else f"X ∈ [{start};{i}]"
        classes.append({"label": label, "obs": int(acc), "p": float(acc_p)})
        i += 1

    if len(classes) > 1 and classes[-1]["p"] * total < 5:
        last = classes.pop()
        classes[-1]["obs"] += last["obs"]
        classes[-1]["p"] += last["p"]
        classes[-1]["label"] += " ∪ " + last["label"]

    return pd.DataFrame(
        {
            "Classe": [c["label"] for c in classes],
            "Observé ni": [c["obs"] for c in classes],
            "p théorique": [round(c["p"], 6) for c in classes],
        }
    )


def compute_chi2_test(obs: np.ndarray, p_th: np.ndarray, r_params: int, alpha: float):
    """Calcule le test d'adéquation du Chi2."""
    N = obs.sum()
    sum_p = p_th.sum()

    norm_factor = 1 / sum_p
    np_i = N * p_th * norm_factor
    chi2_obs = float(np.sum((obs - np_i) ** 2 / np_i))

    ddl = len(obs) - 1 - r_params
    crit = float(chi2.ppf(1 - alpha, ddl))
    pval = float(chi2.sf(chi2_obs, ddl))
    cochran_warning = (np.sum(np_i < 5) / len(obs)) > 0.2

    return {
        "np_i": np_i,
        "chi2_obs": chi2_obs,
        "ddl": ddl,
        "crit": crit,
        "pval": pval,
        "cochran_warning": cochran_warning,
        "rejected": chi2_obs > crit,
    }


def plot_chi2_density(chi2_obs: float, crit: float, ddl: int):
    """Génère la figure de densité du Chi2."""
    x_max = max(chi2_obs, crit) * 1.4 + 1
    x_pts = np.linspace(0.001, x_max, 400)
    y_pts = chi2.pdf(x_pts, ddl)

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.plot(x_pts, y_pts, color="#3f6b4a", linewidth=1.6)
    mask = x_pts >= crit
    ax.fill_between(x_pts[mask], y_pts[mask], color="#9c3b30", alpha=0.2)
    ax.axvline(chi2_obs, color="#9c3b30", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.close(fig)
    return fig
