"""
Steering Vector Decay Experiment v3 — KV Cache Aware

The crucial distinction: when a steering vector modifies hidden states at step t,
those modified hidden states become keys/values in the KV cache. At step t+1,
the attention mechanism attends to these steered KV pairs, potentially propagating
the steering signal even after the vector is no longer applied.

v2 missed this by recomputing full sequences each step (no KV cache persistence).

This version uses incremental generation with past_key_values to properly
capture how the steering effect propagates through the KV cache.

Experiment conditions (all teacher-forced with same token sequence):
1. baseline: No steering, KV cache builds naturally
2. continuous: Steering at every step via KV cache
3. initial_N: Steering for first N steps, KV cache retains steered K/V
4. initial_N_no_cache: Steering for first N steps, but RESET KV cache after
   step N (removing steered K/V). This isolates the token effect from KV effect.
"""

import json
import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda:0"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
STEERING_VECTOR_PATH = "results/steering_vectors/sycophancy_steering_vectors.pt"
DATASET_PATH = "code/CAA/datasets/generate/sycophancy/generate_dataset.json"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

STEER_LAYER = 21
MEASURE_LAYERS = [14, 21, 28]  # Before, at, and after steer layer
ALPHA = 8.0
STEER_DURATIONS = [1, 3, 5, 10]
TOTAL_GEN = 40
NUM_PROMPTS = 60


def incremental_generate(model, tokenizer, input_ids, token_sequence,
                         sv_inject, steer_layer, alpha, steer_duration,
                         measure_layers, sv_normalized, reset_cache_at=None):
    """
    Generate tokens one at a time using KV cache.
    Teacher-forces token_sequence.
    Steers for first steer_duration steps.
    If reset_cache_at is set, recomputes from scratch at that step (removes steered KV).

    Returns dict of layer -> list of projections onto steering direction at each step.
    """
    device = input_ids.device
    projections = {L: [] for L in measure_layers}
    norms_record = {L: [] for L in measure_layers}

    # Initial forward pass on prompt
    past_kv = None
    current_input = input_ids

    # Hooks for capture and steering
    captured = {}
    do_steer = [False]

    hooks = []

    def make_capture(layer_idx):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = h[:, -1, :].detach().float().clone()
        return hook_fn

    def make_steer():
        def hook_fn(module, inp, out):
            if not do_steer[0]:
                return out
            if isinstance(out, tuple):
                h = out[0].clone()
                h[:, -1, :] = h[:, -1, :] + alpha * sv_inject
                return (h,) + out[1:]
            else:
                out = out.clone()
                out[:, -1, :] = out[:, -1, :] + alpha * sv_inject
                return out
        return hook_fn

    for L in measure_layers:
        hooks.append(model.model.layers[L - 1].register_forward_hook(make_capture(L)))
    hooks.append(model.model.layers[steer_layer - 1].register_forward_hook(make_steer()))

    for step in range(len(token_sequence)):
        captured.clear()
        do_steer[0] = (step < steer_duration)

        # Handle cache reset
        if reset_cache_at is not None and step == reset_cache_at:
            # Recompute from scratch with all tokens so far (no steering)
            all_tokens = torch.cat([input_ids] +
                [torch.tensor([[t]], device=device) for t in token_sequence[:step]], dim=1)
            do_steer[0] = False
            with torch.no_grad():
                out = model(all_tokens, use_cache=True)
            past_kv = out.past_key_values
            # Now continue from step with fresh cache
            current_input = torch.tensor([[token_sequence[step]]], device=device)
            do_steer[0] = False  # After reset, no steering
            captured.clear()
            with torch.no_grad():
                out = model(current_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
        else:
            with torch.no_grad():
                out = model(current_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

        # Record projections
        for L in measure_layers:
            if L in captured:
                h = captured[L].squeeze()
                proj = torch.dot(h, sv_normalized[L].to(h.device)).item()
                norm = h.norm().item()
                projections[L].append(proj)
                norms_record[L].append(norm)
            else:
                projections[L].append(0.0)
                norms_record[L].append(0.0)

        current_input = torch.tensor([[token_sequence[step]]], device=device)

    for h in hooks:
        h.remove()

    return projections, norms_record


def free_generate_incremental(model, tokenizer, input_ids,
                              sv_inject, steer_layer, alpha, steer_duration,
                              measure_layers, sv_normalized, total_length):
    """
    Free autoregressive generation with KV cache, steering for first N steps.
    Returns generated tokens, projections, and norms.
    """
    device = input_ids.device
    projections = {L: [] for L in measure_layers}
    generated_ids = []

    captured = {}
    do_steer = [False]
    hooks = []

    def make_capture(layer_idx):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = h[:, -1, :].detach().float().clone()
        return hook_fn

    def make_steer():
        def hook_fn(module, inp, out):
            if not do_steer[0]:
                return out
            if isinstance(out, tuple):
                h = out[0].clone()
                h[:, -1, :] = h[:, -1, :] + alpha * sv_inject
                return (h,) + out[1:]
            else:
                out = out.clone()
                out[:, -1, :] = out[:, -1, :] + alpha * sv_inject
                return out
        return hook_fn

    for L in measure_layers:
        hooks.append(model.model.layers[L - 1].register_forward_hook(make_capture(L)))
    hooks.append(model.model.layers[steer_layer - 1].register_forward_hook(make_steer()))

    past_kv = None
    current_input = input_ids

    for step in range(total_length):
        captured.clear()
        do_steer[0] = (step < steer_duration)

        with torch.no_grad():
            out = model(current_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        for L in measure_layers:
            if L in captured:
                h = captured[L].squeeze()
                proj = torch.dot(h, sv_normalized[L].to(h.device)).item()
                projections[L].append(proj)
            else:
                projections[L].append(0.0)

        current_input = next_token

    for h in hooks:
        h.remove()
    return generated_ids, projections


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    sv_data = torch.load(STEERING_VECTOR_PATH, map_location="cpu", weights_only=True)
    steering_vectors = sv_data["steering_vectors"]

    sv_inject = steering_vectors[STEER_LAYER].to(DEVICE).to(model.dtype)
    sv_normalized = {}
    for L in MEASURE_LAYERS:
        sv = steering_vectors[L].float().to(DEVICE)
        sv_normalized[L] = sv / sv.norm()

    # Load prompts
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    dataset = dataset[300:300 + NUM_PROMPTS]

    # Results storage
    results = {
        # Teacher-forced conditions (comparing to baseline)
        "baseline": {L: [] for L in MEASURE_LAYERS},
        "continuous": {L: [] for L in MEASURE_LAYERS},
    }
    for N in STEER_DURATIONS:
        results[f"initial_{N}"] = {L: [] for L in MEASURE_LAYERS}         # KV cache kept
        results[f"initial_{N}_nocache"] = {L: [] for L in MEASURE_LAYERS}  # KV cache reset
        results[f"free_{N}"] = {L: [] for L in MEASURE_LAYERS}            # Free generation

    results["free_0"] = {L: [] for L in MEASURE_LAYERS}  # Free, no steering
    results["free_all"] = {L: [] for L in MEASURE_LAYERS}  # Free, continuous steering

    example_texts = {}

    for idx, item in enumerate(tqdm(dataset, desc="Running v3 experiments")):
        question = item["question"]
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Step 1: Generate reference token sequence with continuous steering
        cont_ids, cont_projs = free_generate_incremental(
            model, tokenizer, input_ids, sv_inject, STEER_LAYER, ALPHA,
            steer_duration=TOTAL_GEN, measure_layers=MEASURE_LAYERS,
            sv_normalized=sv_normalized, total_length=TOTAL_GEN)

        for L in MEASURE_LAYERS:
            results["free_all"][L].append(cont_projs[L])

        if idx < 3:
            example_texts[f"continuous_{idx}"] = tokenizer.decode(cont_ids)

        # Step 2: Free generation without steering (baseline behavior)
        nosteer_ids, nosteer_projs = free_generate_incremental(
            model, tokenizer, input_ids, sv_inject, STEER_LAYER, ALPHA,
            steer_duration=0, measure_layers=MEASURE_LAYERS,
            sv_normalized=sv_normalized, total_length=TOTAL_GEN)

        for L in MEASURE_LAYERS:
            results["free_0"][L].append(nosteer_projs[L])

        if idx < 3:
            example_texts[f"no_steer_{idx}"] = tokenizer.decode(nosteer_ids)

        # Step 3: Teacher-forced baseline (cont_ids, no steering)
        bl_projs, _ = incremental_generate(
            model, tokenizer, input_ids, cont_ids, sv_inject, STEER_LAYER,
            ALPHA, steer_duration=0, measure_layers=MEASURE_LAYERS,
            sv_normalized=sv_normalized)
        for L in MEASURE_LAYERS:
            results["baseline"][L].append(bl_projs[L])

        # Step 4: Teacher-forced continuous (cont_ids, all steps steered)
        ct_projs, _ = incremental_generate(
            model, tokenizer, input_ids, cont_ids, sv_inject, STEER_LAYER,
            ALPHA, steer_duration=TOTAL_GEN, measure_layers=MEASURE_LAYERS,
            sv_normalized=sv_normalized)
        for L in MEASURE_LAYERS:
            results["continuous"][L].append(ct_projs[L])

        # Step 5: For each N, teacher-forced with initial steering
        for N in STEER_DURATIONS:
            # With KV cache (steered K/V persist)
            tf_projs, _ = incremental_generate(
                model, tokenizer, input_ids, cont_ids, sv_inject, STEER_LAYER,
                ALPHA, steer_duration=N, measure_layers=MEASURE_LAYERS,
                sv_normalized=sv_normalized)
            for L in MEASURE_LAYERS:
                results[f"initial_{N}"][L].append(tf_projs[L])

            # Without steered KV cache (reset at step N)
            nc_projs, _ = incremental_generate(
                model, tokenizer, input_ids, cont_ids, sv_inject, STEER_LAYER,
                ALPHA, steer_duration=N, measure_layers=MEASURE_LAYERS,
                sv_normalized=sv_normalized, reset_cache_at=N)
            for L in MEASURE_LAYERS:
                results[f"initial_{N}_nocache"][L].append(nc_projs[L])

            # Free generation with initial steering
            free_ids, free_projs = free_generate_incremental(
                model, tokenizer, input_ids, sv_inject, STEER_LAYER, ALPHA,
                steer_duration=N, measure_layers=MEASURE_LAYERS,
                sv_normalized=sv_normalized, total_length=TOTAL_GEN)
            for L in MEASURE_LAYERS:
                results[f"free_{N}"][L].append(free_projs[L])

            if idx < 3:
                example_texts[f"free_{N}_{idx}"] = tokenizer.decode(free_ids)

    # Convert to numpy and compute deltas
    save_dict = {}
    for cond_name, layer_data in results.items():
        for L in MEASURE_LAYERS:
            arr = np.array(layer_data[L])
            save_dict[f"{cond_name}_L{L}"] = arr

    # Compute deltas (steered - baseline) for teacher-forced conditions
    for L in MEASURE_LAYERS:
        baseline_arr = np.array(results["baseline"][L])
        for cond_name in (["continuous"] +
                          [f"initial_{N}" for N in STEER_DURATIONS] +
                          [f"initial_{N}_nocache" for N in STEER_DURATIONS]):
            steered_arr = np.array(results[cond_name][L])
            delta = steered_arr - baseline_arr
            save_dict[f"delta_{cond_name}_L{L}"] = delta

    np.savez(os.path.join(RESULTS_DIR, "decay_v3_raw.npz"), **save_dict)

    config_out = {
        "config": {
            "model": MODEL_NAME,
            "steer_layer": STEER_LAYER,
            "measure_layers": MEASURE_LAYERS,
            "alpha": ALPHA,
            "steer_durations": STEER_DURATIONS,
            "total_gen": TOTAL_GEN,
            "num_prompts": NUM_PROMPTS,
            "seed": SEED,
        },
        "example_texts": example_texts,
    }
    with open(os.path.join(RESULTS_DIR, "decay_v3_results.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\nSaved results to {RESULTS_DIR}/decay_v3_*")

    # Quick preview
    for L in MEASURE_LAYERS:
        baseline_arr = np.array(results["baseline"][L])
        print(f"\nLayer {L}:")
        print(f"  Baseline proj mean: {baseline_arr.mean():.4f}")
        cont_arr = np.array(results["continuous"][L])
        delta = cont_arr - baseline_arr
        print(f"  Continuous delta mean: {delta.mean():.4f}")
        for N in STEER_DURATIONS:
            steered_arr = np.array(results[f"initial_{N}"][L])
            d = steered_arr - baseline_arr
            d_mean = d.mean(axis=0)
            print(f"  N={N} delta: peak={d_mean[N-1]:.4f}, post-5avg={d_mean[N:N+5].mean():.4f}, "
                  f"final-5avg={d_mean[-5:].mean():.4f}")


if __name__ == "__main__":
    main()
