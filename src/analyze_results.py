"""
Analyze and visualize steering vector decay results.

Creates publication-quality plots showing:
1. Decay curves for different steering durations
2. Free vs teacher-forced comparison
3. Layer-wise analysis
4. Statistical tests and decay fitting
"""

import json
import numpy as np
import os
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"


def load_results():
    with open(os.path.join(RESULTS_DIR, "decay_results.json"), "r") as f:
        data = json.load(f)
    raw = np.load(os.path.join(RESULTS_DIR, "decay_raw.npz"))
    return data, raw


def exponential_decay(t, A, tau, C):
    """Exponential decay: A * exp(-t/tau) + C"""
    return A * np.exp(-t / tau) + C


def fit_decay(mean_curve, start_idx):
    """Fit exponential decay to the mean curve starting from start_idx."""
    t = np.arange(len(mean_curve) - start_idx)
    y = mean_curve[start_idx:]
    try:
        popt, pcov = curve_fit(
            exponential_decay, t, y,
            p0=[y[0] - y[-1], 5.0, y[-1]],
            maxfev=5000,
            bounds=([0, 0.1, -1], [1, 100, 1])
        )
        y_pred = exponential_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"A": popt[0], "tau": popt[1], "C": popt[2], "R2": r_squared}
    except Exception as e:
        return {"A": 0, "tau": 0, "C": 0, "R2": 0, "error": str(e)}


def plot_main_decay_curves(data, raw):
    """Plot 1: Main decay curves for different N values at the steering layer."""
    config = data["config"]
    steer_layer = config["steer_layer"]
    durations = config["steer_durations"]
    total_gen = config["total_gen"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    t = np.arange(total_gen)

    # No steering baseline
    key = f"no_steering_layer{steer_layer}"
    baseline_mean = raw[key].mean(axis=0)
    baseline_std = raw[key].std(axis=0)
    ax.axhline(y=np.mean(baseline_mean), color="gray", linestyle="--", alpha=0.5,
               label="No steering (mean)")

    # Continuous steering
    key = f"continuous_layer{steer_layer}"
    cont_mean = raw[key].mean(axis=0)
    ax.plot(t, cont_mean, color="black", linewidth=2, label="Continuous steering")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(durations)))

    for i, N in enumerate(durations):
        # Free autoregressive
        key = f"initial_{N}_free_layer{steer_layer}"
        free_mean = raw[key].mean(axis=0)
        free_std = raw[key].std(axis=0)
        ax.plot(t, free_mean, color=colors[i], linewidth=2, label=f"N={N} (free)")
        ax.fill_between(t, free_mean - free_std, free_mean + free_std,
                        color=colors[i], alpha=0.15)

        # Mark where steering stops
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.3)

    ax.set_xlabel("Token Position", fontsize=13)
    ax.set_ylabel(f"Cosine Similarity to Steering Direction (Layer {steer_layer})", fontsize=13)
    ax.set_title("Steering Vector Decay: Free Autoregressive Generation", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "decay_curves_free.png"), dpi=150)
    plt.close()
    print("Saved: decay_curves_free.png")


def plot_free_vs_teacher(data, raw):
    """Plot 2: Free vs teacher-forced comparison for each N."""
    config = data["config"]
    steer_layer = config["steer_layer"]
    durations = config["steer_durations"]
    total_gen = config["total_gen"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    t = np.arange(total_gen)

    for idx, N in enumerate(durations):
        ax = axes[idx]

        # No steering baseline
        baseline_key = f"no_steering_layer{steer_layer}"
        baseline_mean = np.mean(raw[baseline_key].mean(axis=0))
        ax.axhline(y=baseline_mean, color="gray", linestyle="--", alpha=0.5,
                   label="No steering (mean)")

        # Free
        free_key = f"initial_{N}_free_layer{steer_layer}"
        free_mean = raw[free_key].mean(axis=0)
        free_std = raw[free_key].std(axis=0)
        ax.plot(t, free_mean, color="blue", linewidth=2, label="Free generation")
        ax.fill_between(t, free_mean - free_std, free_mean + free_std,
                        color="blue", alpha=0.1)

        # Teacher-forced
        tf_key = f"initial_{N}_teacher_layer{steer_layer}"
        tf_mean = raw[tf_key].mean(axis=0)
        tf_std = raw[tf_key].std(axis=0)
        ax.plot(t, tf_mean, color="red", linewidth=2, label="Teacher-forced")
        ax.fill_between(t, tf_mean - tf_std, tf_mean + tf_std,
                        color="red", alpha=0.1)

        # Continuous
        cont_key = f"continuous_layer{steer_layer}"
        cont_mean = raw[cont_key].mean(axis=0)
        ax.plot(t, cont_mean, color="black", linewidth=1, linestyle="--",
                alpha=0.5, label="Continuous")

        ax.axvline(x=N - 0.5, color="green", linestyle=":", alpha=0.5,
                   label="Steering stops")
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(f"N = {N} steered tokens", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Free vs Teacher-Forced Generation After Initial Steering", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "free_vs_teacher.png"), dpi=150)
    plt.close()
    print("Saved: free_vs_teacher.png")


def plot_layer_comparison(data, raw):
    """Plot 3: How decay varies across layers."""
    config = data["config"]
    measure_layers = config["measure_layers"]
    N = 3  # Use N=3 as representative

    fig, axes = plt.subplots(1, len(measure_layers), figsize=(5 * len(measure_layers), 5))
    total_gen = config["total_gen"]
    t = np.arange(total_gen)

    for idx, layer in enumerate(measure_layers):
        ax = axes[idx]

        # Free
        free_key = f"initial_{N}_free_layer{layer}"
        free_mean = raw[free_key].mean(axis=0)
        ax.plot(t, free_mean, color="blue", linewidth=2, label="Free")

        # Teacher-forced
        tf_key = f"initial_{N}_teacher_layer{layer}"
        tf_mean = raw[tf_key].mean(axis=0)
        ax.plot(t, tf_mean, color="red", linewidth=2, label="Teacher-forced")

        # Continuous
        cont_key = f"continuous_layer{layer}"
        cont_mean = raw[cont_key].mean(axis=0)
        ax.plot(t, cont_mean, color="black", linewidth=1, linestyle="--",
                alpha=0.5, label="Continuous")

        # Baseline
        base_key = f"no_steering_layer{layer}"
        ax.axhline(y=np.mean(raw[base_key].mean(axis=0)), color="gray",
                   linestyle="--", alpha=0.5)

        ax.axvline(x=N - 0.5, color="green", linestyle=":", alpha=0.5)
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(f"Layer {layer}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Layer-wise Decay (N={N} steered tokens)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "layer_comparison.png"), dpi=150)
    plt.close()
    print("Saved: layer_comparison.png")


def compute_statistics(data, raw):
    """Compute statistical tests and decay fits."""
    config = data["config"]
    steer_layer = config["steer_layer"]
    durations = config["steer_durations"]
    total_gen = config["total_gen"]

    stats_results = {}

    for N in durations:
        free_key = f"initial_{N}_free_layer{steer_layer}"
        tf_key = f"initial_{N}_teacher_layer{steer_layer}"

        free_data = raw[free_key]  # (num_prompts, total_gen)
        tf_data = raw[tf_key]

        # Compare post-steering mean similarity
        post_start = N
        post_end = min(N + 20, total_gen)

        free_post_means = free_data[:, post_start:post_end].mean(axis=1)
        tf_post_means = tf_data[:, post_start:post_end].mean(axis=1)

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(tf_post_means, free_post_means)
        effect_size = (tf_post_means.mean() - free_post_means.mean()) / np.sqrt(
            (tf_post_means.std() ** 2 + free_post_means.std() ** 2) / 2
        )

        # Fit exponential decay to free condition
        free_mean = free_data.mean(axis=0)
        tf_mean = tf_data.mean(axis=0)

        free_fit = fit_decay(free_mean, N)
        tf_fit = fit_decay(tf_mean, N)

        stats_results[f"N={N}"] = {
            "free_post_mean": float(free_post_means.mean()),
            "free_post_std": float(free_post_means.std()),
            "teacher_post_mean": float(tf_post_means.mean()),
            "teacher_post_std": float(tf_post_means.std()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(effect_size),
            "free_decay_fit": free_fit,
            "teacher_decay_fit": tf_fit,
        }

    # Baseline statistics
    base_key = f"no_steering_layer{steer_layer}"
    cont_key = f"continuous_layer{steer_layer}"
    stats_results["baseline"] = {
        "no_steering_mean": float(raw[base_key].mean()),
        "no_steering_std": float(raw[base_key].std()),
        "continuous_mean": float(raw[cont_key].mean()),
        "continuous_std": float(raw[cont_key].std()),
    }

    return stats_results


def plot_decay_fits(data, raw):
    """Plot 4: Exponential decay fits."""
    config = data["config"]
    steer_layer = config["steer_layer"]
    durations = config["steer_durations"]
    total_gen = config["total_gen"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors_free = plt.cm.Blues(np.linspace(0.4, 0.9, len(durations)))
    colors_tf = plt.cm.Reds(np.linspace(0.4, 0.9, len(durations)))

    fit_results = []

    for i, N in enumerate(durations):
        free_key = f"initial_{N}_free_layer{steer_layer}"
        tf_key = f"initial_{N}_teacher_layer{steer_layer}"

        free_mean = raw[free_key].mean(axis=0)
        tf_mean = raw[tf_key].mean(axis=0)

        t_post = np.arange(total_gen - N)

        # Plot data
        ax.plot(t_post, free_mean[N:], color=colors_free[i], linewidth=1.5,
                label=f"N={N} free", alpha=0.8)
        ax.plot(t_post, tf_mean[N:], color=colors_tf[i], linewidth=1.5,
                linestyle="--", label=f"N={N} teacher", alpha=0.8)

        # Fit and plot decay
        free_fit = fit_decay(free_mean, N)
        tf_fit = fit_decay(tf_mean, N)

        if free_fit.get("R2", 0) > 0.1:
            y_fit = exponential_decay(t_post, free_fit["A"], free_fit["tau"], free_fit["C"])
            ax.plot(t_post, y_fit, color=colors_free[i], linewidth=1, linestyle=":",
                    alpha=0.5)

        fit_results.append({
            "N": N,
            "free_tau": free_fit.get("tau", 0),
            "free_R2": free_fit.get("R2", 0),
            "teacher_tau": tf_fit.get("tau", 0),
            "teacher_R2": tf_fit.get("R2", 0),
        })

    ax.set_xlabel("Tokens After Steering Stops", fontsize=13)
    ax.set_ylabel(f"Cosine Similarity to Steering Direction (Layer {steer_layer})", fontsize=13)
    ax.set_title("Post-Steering Decay: Free vs Teacher-Forced", fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "decay_fits.png"), dpi=150)
    plt.close()
    print("Saved: decay_fits.png")

    return fit_results


def plot_half_life_comparison(data, raw):
    """Plot 5: Bar chart of decay time constants."""
    config = data["config"]
    steer_layer = config["steer_layer"]
    durations = config["steer_durations"]

    free_taus = []
    tf_taus = []

    for N in durations:
        free_key = f"initial_{N}_free_layer{steer_layer}"
        tf_key = f"initial_{N}_teacher_layer{steer_layer}"

        free_mean = raw[free_key].mean(axis=0)
        tf_mean = raw[tf_key].mean(axis=0)

        free_fit = fit_decay(free_mean, N)
        tf_fit = fit_decay(tf_mean, N)

        free_taus.append(free_fit.get("tau", 0))
        tf_taus.append(tf_fit.get("tau", 0))

    x = np.arange(len(durations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, free_taus, width, label="Free generation",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, tf_taus, width, label="Teacher-forced",
                   color="indianred", alpha=0.8)

    ax.set_xlabel("Number of Steered Tokens (N)", fontsize=13)
    ax.set_ylabel("Decay Time Constant (τ, tokens)", fontsize=13)
    ax.set_title("Decay Speed: Free vs Teacher-Forced", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in durations])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "half_life_comparison.png"), dpi=150)
    plt.close()
    print("Saved: half_life_comparison.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    data, raw = load_results()

    print("=" * 60)
    print("ANALYSIS OF STEERING VECTOR DECAY")
    print("=" * 60)

    # Generate all plots
    plot_main_decay_curves(data, raw)
    plot_free_vs_teacher(data, raw)
    plot_layer_comparison(data, raw)
    fit_results = plot_decay_fits(data, raw)
    plot_half_life_comparison(data, raw)

    # Compute statistics
    stats_results = compute_statistics(data, raw)

    print("\n" + "=" * 60)
    print("STATISTICAL RESULTS")
    print("=" * 60)

    print(f"\nBaseline (no steering): mean sim = {stats_results['baseline']['no_steering_mean']:.4f}")
    print(f"Continuous steering:    mean sim = {stats_results['baseline']['continuous_mean']:.4f}")

    for N in data["config"]["steer_durations"]:
        key = f"N={N}"
        r = stats_results[key]
        print(f"\n--- N = {N} steered tokens ---")
        print(f"  Free post-steering mean sim:     {r['free_post_mean']:.4f} ± {r['free_post_std']:.4f}")
        print(f"  Teacher post-steering mean sim:   {r['teacher_post_mean']:.4f} ± {r['teacher_post_std']:.4f}")
        print(f"  Paired t-test: t={r['t_statistic']:.3f}, p={r['p_value']:.4f}")
        print(f"  Cohen's d: {r['cohens_d']:.3f}")
        print(f"  Free decay τ: {r['free_decay_fit'].get('tau', 'N/A'):.2f} (R²={r['free_decay_fit'].get('R2', 0):.3f})")
        print(f"  Teacher decay τ: {r['teacher_decay_fit'].get('tau', 'N/A'):.2f} (R²={r['teacher_decay_fit'].get('R2', 0):.3f})")

    # Save statistics
    with open(os.path.join(RESULTS_DIR, "statistics.json"), "w") as f:
        json.dump(stats_results, f, indent=2, default=str)

    # Print example generations
    print("\n" + "=" * 60)
    print("EXAMPLE GENERATIONS")
    print("=" * 60)
    for key in sorted(data["example_texts"].keys()):
        if "0" in key:  # Only show first prompt
            print(f"\n--- {key} ---")
            print(data["example_texts"][key][:200])

    print("\n" + "=" * 60)
    print("DECAY FIT SUMMARY")
    print("=" * 60)
    print(f"{'N':>3} | {'Free τ':>8} | {'Free R²':>8} | {'Teacher τ':>10} | {'Teacher R²':>10}")
    print("-" * 50)
    for r in fit_results:
        print(f"{r['N']:>3} | {r['free_tau']:>8.2f} | {r['free_R2']:>8.3f} | {r['teacher_tau']:>10.2f} | {r['teacher_R2']:>10.3f}")


if __name__ == "__main__":
    main()
