# Steering Vector Decay

Empirical study of how quickly activation steering influence decays in transformer hidden states after the steering vector is no longer applied.

## Key Findings

- **Steering decays to <5% of peak in 1 token** after the vector is removed (teacher-forced, KV cache preserved)
- **Exponential decay** with time constants τ ≈ 1.2–5.8 tokens (R² up to 0.97)
- **KV cache is the persistence mechanism**: preserves a small positive trace; resetting it causes overcorrection
- **More steered tokens → faster relative decay** (counterintuitive: stronger perturbation triggers stronger correction)
- **Continuous steering is necessary** — initial-only steering is essentially ineffective

## Model & Data

- **Model**: Qwen/Qwen2.5-1.5B-Instruct (28 layers)
- **Steering**: CAA sycophancy vectors, α=8.0 at layer 21
- **Measurement**: Δ projection of (steered − baseline) hidden states onto steering direction at layer 28
- **Dataset**: 300 pairs for extraction, 60 held-out prompts for evaluation

## How to Reproduce

```bash
# Setup
uv venv && uv pip install torch transformers accelerate numpy matplotlib scipy tqdm

# Extract steering vectors
python src/extract_steering_vectors.py

# Run decay experiment (v3 — KV-cache-aware, ~16 min on A6000)
python src/decay_experiment_v3.py

# Analyze and plot
python src/final_analysis.py
```

## File Structure

```
├── REPORT.md                    # Full research report
├── planning.md                  # Experimental plan
├── src/
│   ├── extract_steering_vectors.py   # CAA vector extraction
│   ├── decay_experiment_v3.py        # Main experiment (KV cache aware)
│   └── final_analysis.py            # Analysis and plotting
├── results/
│   ├── steering_vectors/             # Extracted CAA vectors
│   ├── decay_v3_raw.npz             # Raw experimental data
│   ├── final_statistics.json         # Statistical summary
│   └── plots/                        # All figures
├── code/                             # Cloned reference repos
├── datasets/                         # Downloaded datasets
├── papers/                           # Reference papers
└── literature_review.md              # Literature synthesis
```

## Citation

This work builds on Contrastive Activation Addition (Rimsky et al., 2024) and is informed by Dynamic Activation Composition (Scalena et al., 2024) and KV Cache Steering (Belitsky et al., 2025). See REPORT.md for full references.
