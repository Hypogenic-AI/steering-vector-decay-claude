"""
Final analysis and publication-quality plots for Steering Vector Decay.
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
L = 28  # Layer with the signal (after steering layer 21)


def exponential_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def fit_decay(y):
    t = np.arange(len(y))
    try:
        popt, _ = curve_fit(exponential_decay, t, y,
                            p0=[y[0], 2.0, 0], maxfev=10000,
                            bounds=([-np.inf, 0.01, -np.inf], [np.inf, 100, np.inf]))
        y_pred = exponential_decay(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"A": popt[0], "tau": popt[1], "C": popt[2], "R2": r2}
    except:
        return None


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    raw = np.load(os.path.join(RESULTS_DIR, "decay_v3_raw.npz"))
    with open(os.path.join(RESULTS_DIR, "decay_v3_results.json")) as f:
        cfg_data = json.load(f)

    cfg = cfg_data["config"]
    DURATIONS = cfg["steer_durations"]
    TOTAL_GEN = cfg["total_gen"]
    t = np.arange(TOTAL_GEN)

    # ================================================================
    # FIGURE 1: Main decay curves (teacher-forced, delta projection)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    # Continuous steering (ceiling)
    cont = raw[f"delta_continuous_L{L}"]
    cont_mean = cont.mean(axis=0)
    cont_se = cont.std(axis=0) / np.sqrt(cont.shape[0])
    ax.plot(t, cont_mean, "k-", linewidth=2.5, label="Continuous steering", zorder=10)
    ax.fill_between(t, cont_mean - 1.96 * cont_se, cont_mean + 1.96 * cont_se,
                    color="gray", alpha=0.2)

    for i, N in enumerate(DURATIONS):
        key = f"delta_initial_{N}_L{L}"
        data = raw[key]
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])

        ax.plot(t, mean, color=colors[i], linewidth=2, label=f"Steer first {N} tokens")
        ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=colors[i], alpha=0.15)
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.4, linewidth=1)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Token Position (generation step)", fontsize=14)
    ax.set_ylabel("Δ Projection onto Steering Direction", fontsize=14)
    ax.set_title("Steering Vector Decay in Hidden States\n"
                 "(Teacher-Forced, KV Cache Preserved, Layer 28 readout)", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_main_decay.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig1_main_decay.png")

    # ================================================================
    # FIGURE 2: Zoomed post-steering decay (log scale if useful)
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fit_summary = []

    for i, N in enumerate(DURATIONS):
        ax = axes[i]
        key = f"delta_initial_{N}_L{L}"
        data = raw[key]
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])

        # Post-steering portion
        post_mean = mean[N:]
        post_se = se[N:]
        t_post = np.arange(len(post_mean))

        ax.plot(t_post, post_mean, color=colors[i], linewidth=2, marker="o",
                markersize=3, label=f"N={N} (data)")
        ax.fill_between(t_post, post_mean - 1.96 * post_se, post_mean + 1.96 * post_se,
                        color=colors[i], alpha=0.15)

        # Fit exponential decay
        fit = fit_decay(post_mean)
        if fit and fit["R2"] > 0.1:
            y_fit = exponential_decay(t_post, fit["A"], fit["tau"], fit["C"])
            ax.plot(t_post, y_fit, "k--", linewidth=1.5,
                    label=f"Exp fit: τ={fit['tau']:.2f}, R²={fit['R2']:.3f}")
            fit_summary.append({"N": N, **fit})
        else:
            fit_summary.append({"N": N, "tau": None, "R2": 0, "note": "fit failed"})

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Tokens after steering stops", fontsize=11)
        ax.set_ylabel("Δ Projection", fontsize=11)
        ax.set_title(f"N = {N} steered tokens", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Post-Steering Decay with Exponential Fits", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_decay_fits.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig2_decay_fits.png")

    # ================================================================
    # FIGURE 3: KV cache vs no KV cache comparison
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, N in enumerate(DURATIONS):
        ax = axes[i]

        # With KV cache
        key_kv = f"delta_initial_{N}_L{L}"
        kv_mean = raw[key_kv].mean(axis=0)
        kv_se = raw[key_kv].std(axis=0) / np.sqrt(raw[key_kv].shape[0])

        # Without KV cache (reset)
        key_nc = f"delta_initial_{N}_nocache_L{L}"
        nc_mean = raw[key_nc].mean(axis=0)
        nc_se = raw[key_nc].std(axis=0) / np.sqrt(raw[key_nc].shape[0])

        ax.plot(t, kv_mean, color="steelblue", linewidth=2, label="With KV cache")
        ax.fill_between(t, kv_mean - 1.96 * kv_se, kv_mean + 1.96 * kv_se,
                        color="steelblue", alpha=0.15)

        ax.plot(t, nc_mean, color="indianred", linewidth=2, label="KV cache reset")
        ax.fill_between(t, nc_mean - 1.96 * nc_se, nc_mean + 1.96 * nc_se,
                        color="indianred", alpha=0.15)

        ax.axvline(x=N - 0.5, color="green", linestyle=":", alpha=0.5, label="Steering stops")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_ylabel("Δ Projection", fontsize=11)
        ax.set_title(f"N = {N} steered tokens", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Effect of KV Cache on Steering Persistence\n"
                 "(Teacher-Forced, Same Tokens)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_kv_cache_effect.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig3_kv_cache_effect.png")

    # ================================================================
    # FIGURE 4: Free generation projections
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # No steering free gen
    ns = raw[f"free_0_L{L}"]
    ns_mean = ns.mean(axis=0)
    ax.plot(t, ns_mean, "k--", linewidth=1.5, alpha=0.6, label="No steering (free)")

    # Continuous free gen
    ca = raw[f"free_all_L{L}"]
    ca_mean = ca.mean(axis=0)
    ax.plot(t, ca_mean, "k-", linewidth=2, label="Continuous steering (free)")

    for i, N in enumerate(DURATIONS):
        key = f"free_{N}_L{L}"
        data = raw[key]
        mean = data.mean(axis=0)
        se = data.std(axis=0) / np.sqrt(data.shape[0])
        ax.plot(t, mean, color=colors[i], linewidth=2, label=f"Steer first {N} (free)")
        ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=colors[i], alpha=0.1)
        ax.axvline(x=N - 0.5, color=colors[i], linestyle=":", alpha=0.3)

    ax.set_xlabel("Token Position", fontsize=14)
    ax.set_ylabel("Projection onto Steering Direction", fontsize=14)
    ax.set_title("Free Autoregressive Generation: Raw Projection", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_free_generation.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig4_free_generation.png")

    # ================================================================
    # FIGURE 5: Normalized decay comparison
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, N in enumerate(DURATIONS):
        key = f"delta_initial_{N}_L{L}"
        mean = raw[key].mean(axis=0)
        peak = mean[N - 1] if abs(mean[N - 1]) > 0.01 else 1
        post = mean[N:N + 20]
        normalized = post / abs(peak)
        t_post = np.arange(len(normalized))
        ax.plot(t_post, normalized, color=colors[i], linewidth=2, marker="o",
                markersize=4, label=f"N={N}")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, label="50% level")
    ax.axhline(y=0.1, color="gray", linestyle=":", alpha=0.2, label="10% level")
    ax.set_xlabel("Tokens After Steering Stops", fontsize=14)
    ax.set_ylabel("Normalized Δ (fraction of peak)", fontsize=14)
    ax.set_title("Normalized Steering Decay Across Conditions", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_normalized_decay.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig5_normalized_decay.png")

    # ================================================================
    # Statistical Analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL STATISTICAL ANALYSIS")
    print("=" * 70)

    all_stats = {}

    for N in DURATIONS:
        key = f"delta_initial_{N}_L{L}"
        data = raw[key]
        mean = data.mean(axis=0)
        peak = mean[N - 1]

        # How many tokens until delta < 10% of peak?
        tokens_to_10pct = None
        for j in range(N, len(mean)):
            if abs(mean[j]) < abs(peak) * 0.1:
                tokens_to_10pct = j - N
                break

        # How many tokens until delta < 50% of peak?
        tokens_to_50pct = None
        for j in range(N, len(mean)):
            if abs(mean[j]) < abs(peak) * 0.5:
                tokens_to_50pct = j - N
                break

        # Post-steering mean (first 5 steps)
        post5 = mean[N:N + 5].mean()
        post10 = mean[N:N + 10].mean()

        # Paired test: KV cache vs no KV cache post-steering
        kv_data = raw[f"delta_initial_{N}_L{L}"]
        nc_data = raw[f"delta_initial_{N}_nocache_L{L}"]
        kv_post = kv_data[:, N:N + 10].mean(axis=1)
        nc_post = nc_data[:, N:N + 10].mean(axis=1)
        t_stat, p_val = stats.ttest_rel(kv_post, nc_post)
        d = (kv_post.mean() - nc_post.mean()) / np.sqrt(
            (kv_post.std() ** 2 + nc_post.std() ** 2) / 2)

        r = {
            "peak_delta": float(peak),
            "tokens_to_50pct": tokens_to_50pct,
            "tokens_to_10pct": tokens_to_10pct,
            "post5_mean": float(post5),
            "post10_mean": float(post10),
            "kv_vs_nocache_t": float(t_stat),
            "kv_vs_nocache_p": float(p_val),
            "kv_vs_nocache_d": float(d),
        }

        fit = fit_decay(mean[N:])
        if fit:
            r["exp_fit"] = fit

        all_stats[f"N={N}"] = r

        print(f"\n--- N = {N} steered tokens ---")
        print(f"  Peak Δ projection:        {peak:.3f}")
        print(f"  Tokens to 50% of peak:    {tokens_to_50pct if tokens_to_50pct else '>30'}")
        print(f"  Tokens to 10% of peak:    {tokens_to_10pct if tokens_to_10pct else '>30'}")
        print(f"  Post-steering 5-step avg: {post5:.4f} ({post5/abs(peak)*100:.1f}% of peak)")
        print(f"  Post-steering 10-step avg:{post10:.4f} ({post10/abs(peak)*100:.1f}% of peak)")
        print(f"  KV vs no-cache: t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")
        if fit and fit["R2"] > 0.1:
            print(f"  Exp fit: τ={fit['tau']:.3f}, A={fit['A']:.3f}, C={fit['C']:.4f}, R²={fit['R2']:.3f}")

    # Continuous reference
    cont = raw[f"delta_continuous_L{L}"]
    cont_mean_val = cont.mean(axis=0).mean()
    all_stats["continuous_mean"] = float(cont_mean_val)
    print(f"\nContinuous steering mean Δ: {cont_mean_val:.3f}")

    # Save
    with open(os.path.join(RESULTS_DIR, "final_statistics.json"), "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    print(f"\nSaved: {RESULTS_DIR}/final_statistics.json")

    # Print example texts
    print("\n" + "=" * 70)
    print("EXAMPLE GENERATIONS (first prompt)")
    print("=" * 70)
    for key in sorted(cfg_data["example_texts"]):
        if key.endswith("_0"):
            text = cfg_data["example_texts"][key]
            print(f"\n--- {key} ---")
            print(text[:200])


if __name__ == "__main__":
    main()
