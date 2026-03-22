"""
Analyze v2 decay experiment results.
Focus on delta-based metrics: the difference in hidden states between
steered and unsteered conditions, projected onto the steering direction.
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


def exponential_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def fit_decay_safe(y, start=0):
    """Fit exponential decay, return params or None."""
    t = np.arange(len(y) - start)
    y = y[start:]
    if len(t) < 3 or np.std(y) < 1e-10:
        return None
    try:
        popt, _ = curve_fit(
            exponential_decay, t, y,
            p0=[y[0] - y[-1], 5.0, y[-1]],
            maxfev=10000,
            bounds=([-np.inf, 0.1, -np.inf], [np.inf, 200, np.inf])
        )
        y_pred = exponential_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"A": popt[0], "tau": popt[1], "C": popt[2], "R2": r2}
    except:
        return None


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "decay_v2_results.json")) as f:
        config_data = json.load(f)
    raw = np.load(os.path.join(RESULTS_DIR, "decay_v2_raw.npz"))

    cfg = config_data["config"]
    STEER_LAYER = cfg["steer_layer"]
    MEASURE_LAYERS = cfg["measure_layers"]
    DURATIONS = cfg["steer_durations"]
    TOTAL_GEN = cfg["total_gen"]

    print("=" * 70)
    print("STEERING VECTOR DECAY ANALYSIS (v2 - Delta-Based)")
    print("=" * 70)

    # ================================================================
    # PLOT 1: Delta projection decay at steering layer for each N
    # (Teacher-forced condition — cleanest measurement)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    t = np.arange(TOTAL_GEN)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(DURATIONS)))

    # Continuous steering delta (ceiling)
    cont_key = f"cont_delta_proj_L{STEER_LAYER}"
    cont_mean = raw[cont_key].mean(axis=0)
    cont_std = raw[cont_key].std(axis=0) / np.sqrt(raw[cont_key].shape[0])
    ax.plot(t, cont_mean, "k-", linewidth=2.5, label="Continuous steering", zorder=10)
    ax.fill_between(t, cont_mean - 1.96 * cont_std, cont_mean + 1.96 * cont_std,
                    color="gray", alpha=0.2)

    fit_results_tf = {}
    for i, N in enumerate(DURATIONS):
        key = f"tf_delta_proj_N{N}_L{STEER_LAYER}"
        data = raw[key]  # (num_prompts, total_gen)
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])

        ax.plot(t, mean, color=colors[i], linewidth=2, label=f"N={N} (teacher-forced)")
        ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=colors[i], alpha=0.15)
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.4)

        # Fit decay after steering stops
        fit = fit_decay_safe(mean, start=N)
        fit_results_tf[N] = fit

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Token Position", fontsize=13)
    ax.set_ylabel("Δ Projection onto Steering Direction", fontsize=13)
    ax.set_title("Steering Signal Decay in Hidden States (Teacher-Forced, Same Tokens)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_delta_projection_decay.png"), dpi=150)
    plt.close()
    print("Saved: v2_delta_projection_decay.png")

    # ================================================================
    # PLOT 2: Delta norm decay (how much do hidden states differ overall?)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, N in enumerate(DURATIONS):
        key = f"tf_delta_norm_N{N}_L{STEER_LAYER}"
        data = raw[key]
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])

        ax.plot(t, mean, color=colors[i], linewidth=2, label=f"N={N}")
        ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=colors[i], alpha=0.15)
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.4)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Token Position", fontsize=13)
    ax.set_ylabel("||Δh|| (L2 Norm of Hidden State Difference)", fontsize=13)
    ax.set_title("Total Magnitude of Steering Effect Over Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_delta_norm_decay.png"), dpi=150)
    plt.close()
    print("Saved: v2_delta_norm_decay.png")

    # ================================================================
    # PLOT 3: Delta cosine similarity (is the delta still aligned with v?)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, N in enumerate(DURATIONS):
        key = f"tf_delta_cos_N{N}_L{STEER_LAYER}"
        data = raw[key]
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])

        ax.plot(t, mean, color=colors[i], linewidth=2, label=f"N={N}")
        ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=colors[i], alpha=0.15)
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.4)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Token Position", fontsize=13)
    ax.set_ylabel("cos(Δh, v) — Alignment of Perturbation with Steering Direction", fontsize=13)
    ax.set_title("Direction of Residual Steering Effect", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_delta_cosine_decay.png"), dpi=150)
    plt.close()
    print("Saved: v2_delta_cosine_decay.png")

    # ================================================================
    # PLOT 4: Layer comparison — delta projection at different layers
    # ================================================================
    N_rep = 3  # Representative N value
    fig, axes = plt.subplots(1, len(MEASURE_LAYERS), figsize=(5 * len(MEASURE_LAYERS), 5))

    for idx, L in enumerate(MEASURE_LAYERS):
        ax = axes[idx]

        # Continuous
        cont_key = f"cont_delta_proj_L{L}"
        if cont_key in raw:
            ax.plot(t, raw[cont_key].mean(axis=0), "k-", linewidth=1.5,
                    label="Continuous", alpha=0.6)

        for i, N in enumerate(DURATIONS):
            key = f"tf_delta_proj_N{N}_L{L}"
            mean = raw[key].mean(axis=0)
            ax.plot(t, mean, color=colors[i], linewidth=1.5, label=f"N={N}")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_ylabel("Δ Projection", fontsize=11)
        ax.set_title(f"Layer {L}", fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Steering Decay Across Layers (Teacher-Forced)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_layer_comparison.png"), dpi=150)
    plt.close()
    print("Saved: v2_layer_comparison.png")

    # ================================================================
    # PLOT 5: Free generation projections (not delta, just raw projection)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # No-steer baseline
    no_steer_key = f"no_steer_proj_L{STEER_LAYER}"
    ns_mean = raw[no_steer_key].mean(axis=0)
    ax.plot(t, ns_mean, "k--", linewidth=1, alpha=0.5, label="No steering (baseline)")

    # Continuous
    cont_key = f"cont_proj_L{STEER_LAYER}"
    c_mean = raw[cont_key].mean(axis=0)
    ax.plot(t, c_mean, "k-", linewidth=2, label="Continuous steering")

    for i, N in enumerate(DURATIONS):
        key = f"free_proj_N{N}_L{STEER_LAYER}"
        mean = raw[key].mean(axis=0)
        se = raw[key].std(axis=0) / np.sqrt(raw[key].shape[0])
        ax.plot(t, mean, color=colors[i], linewidth=2, label=f"N={N} (free)")
        ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=colors[i], alpha=0.1)
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.3)

    ax.set_xlabel("Token Position", fontsize=13)
    ax.set_ylabel(f"Projection onto Steering Direction (Layer {STEER_LAYER})", fontsize=13)
    ax.set_title("Free Generation: Raw Projection onto Steering Direction", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_free_projection.png"), dpi=150)
    plt.close()
    print("Saved: v2_free_projection.png")

    # ================================================================
    # PLOT 6: Summary — Normalized decay curves with exponential fits
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Teacher-forced delta projection, normalized
    ax = axes[0]
    for i, N in enumerate(DURATIONS):
        key = f"tf_delta_proj_N{N}_L{STEER_LAYER}"
        mean = raw[key].mean(axis=0)
        # Normalize by value at step N-1 (peak during steering)
        peak = mean[N - 1] if abs(mean[N - 1]) > 1e-8 else 1
        normalized = mean[N:] / peak
        t_post = np.arange(len(normalized))
        ax.plot(t_post, normalized, color=colors[i], linewidth=2, label=f"N={N}")

        # Fit
        fit = fit_decay_safe(mean, start=N)
        if fit and fit["R2"] > 0.3:
            y_fit = exponential_decay(t_post, fit["A"], fit["tau"], fit["C"])
            ax.plot(t_post, y_fit / peak, color=colors[i], linewidth=1, linestyle=":")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Tokens After Steering Stops", fontsize=12)
    ax.set_ylabel("Normalized Δ Projection", fontsize=12)
    ax.set_title("Teacher-Forced: Normalized Decay", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Delta norm, normalized
    ax = axes[1]
    for i, N in enumerate(DURATIONS):
        key = f"tf_delta_norm_N{N}_L{STEER_LAYER}"
        mean = raw[key].mean(axis=0)
        peak = mean[N - 1] if mean[N - 1] > 1e-8 else 1
        normalized = mean[N:] / peak
        t_post = np.arange(len(normalized))
        ax.plot(t_post, normalized, color=colors[i], linewidth=2, label=f"N={N}")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Tokens After Steering Stops", fontsize=12)
    ax.set_ylabel("Normalized ||Δh||", fontsize=12)
    ax.set_title("Norm of Perturbation: Normalized Decay", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_normalized_decay.png"), dpi=150)
    plt.close()
    print("Saved: v2_normalized_decay.png")

    # ================================================================
    # PLOT 7: Decay time constants across layers (bar chart)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    layer_tau = {}
    for L in MEASURE_LAYERS:
        taus = []
        for N in DURATIONS:
            key = f"tf_delta_proj_N{N}_L{L}"
            mean = raw[key].mean(axis=0)
            fit = fit_decay_safe(mean, start=N)
            if fit and fit["R2"] > 0.2:
                taus.append(fit["tau"])
            else:
                taus.append(0)
        layer_tau[L] = taus

    x = np.arange(len(DURATIONS))
    width = 0.8 / len(MEASURE_LAYERS)
    layer_colors = plt.cm.Set2(np.linspace(0, 0.8, len(MEASURE_LAYERS)))

    for j, L in enumerate(MEASURE_LAYERS):
        offset = (j - len(MEASURE_LAYERS) / 2 + 0.5) * width
        ax.bar(x + offset, layer_tau[L], width, label=f"Layer {L}",
               color=layer_colors[j], alpha=0.8)

    ax.set_xlabel("Number of Steered Tokens (N)", fontsize=13)
    ax.set_ylabel("Decay Time Constant τ (tokens)", fontsize=13)
    ax.set_title("Decay Speed by Layer and Steering Duration", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in DURATIONS])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "v2_tau_by_layer.png"), dpi=150)
    plt.close()
    print("Saved: v2_tau_by_layer.png")

    # ================================================================
    # Statistical summary
    # ================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    stats_output = {}

    for N in DURATIONS:
        key = f"tf_delta_proj_N{N}_L{STEER_LAYER}"
        data = raw[key]
        mean = data.mean(axis=0)

        # Peak value (at step N-1, last steered position)
        peak = mean[N - 1]

        # Post-steering values
        post_mean = mean[N:N + 10].mean()
        final_mean = mean[-5:].mean()

        # Decay fit
        fit = fit_decay_safe(mean, start=N)

        # Half-life: tokens until delta drops to 50% of peak
        half_life = None
        for t_idx in range(N, len(mean)):
            if abs(mean[t_idx]) < abs(peak) * 0.5:
                half_life = t_idx - N
                break

        print(f"\n--- N = {N} steered tokens (Layer {STEER_LAYER}) ---")
        print(f"  Peak Δ projection (at step {N-1}):  {peak:.4f}")
        print(f"  Post-steering mean (steps {N}-{N+10}): {post_mean:.4f}")
        print(f"  Final mean (last 5 steps):         {final_mean:.4f}")
        print(f"  Half-life:                         {half_life if half_life else '>30'} tokens")
        if fit:
            print(f"  Exp fit: A={fit['A']:.4f}, τ={fit['tau']:.2f}, C={fit['C']:.4f}, R²={fit['R2']:.3f}")
        else:
            print(f"  Exp fit: FAILED (data may not follow exponential decay)")

        stats_output[f"N={N}"] = {
            "peak_delta": float(peak),
            "post_mean": float(post_mean),
            "final_mean": float(final_mean),
            "half_life": half_life,
            "fit": fit if fit else {"error": "fit_failed"},
        }

    # Layer comparison summary
    print("\n" + "=" * 70)
    print("LAYER COMPARISON (N=3, teacher-forced)")
    print("=" * 70)
    N = 3
    for L in MEASURE_LAYERS:
        key = f"tf_delta_proj_N{N}_L{L}"
        mean = raw[key].mean(axis=0)
        peak = mean[N - 1]
        fit = fit_decay_safe(mean, start=N)
        tau_str = f"{fit['tau']:.2f}" if fit and fit["R2"] > 0.2 else "N/A"
        r2_str = f"{fit['R2']:.3f}" if fit else "N/A"
        print(f"  Layer {L:>2}: peak={peak:>8.4f}, τ={tau_str:>6}, R²={r2_str}")

    # Continuous steering for reference
    print("\n--- Continuous steering reference ---")
    for L in MEASURE_LAYERS:
        cont_key = f"cont_delta_proj_L{L}"
        if cont_key in raw:
            mean = raw[cont_key].mean(axis=0)
            print(f"  Layer {L:>2}: mean Δ proj = {mean.mean():.4f}, std = {mean.std():.4f}")

    # Save stats
    with open(os.path.join(RESULTS_DIR, "statistics_v2.json"), "w") as f:
        json.dump(stats_output, f, indent=2, default=str)

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print(f"Statistics saved to {RESULTS_DIR}/statistics_v2.json")


if __name__ == "__main__":
    main()
