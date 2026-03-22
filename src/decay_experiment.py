"""
Steering Vector Decay Experiment

Core experiment: Apply steering vector for first N tokens of generation,
then measure how quickly hidden state similarity to the steering direction
decays in subsequent tokens.

Two conditions:
1. Free autoregressive: Model generates freely after steering stops
2. Teacher-forced: Feed the tokens generated WITH steering, but WITHOUT
   adding the steering vector to hidden states

Measures cosine similarity of hidden states to steering direction at each position.
"""

import json
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
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

# Experiment parameters
STEER_LAYER = 21          # Layer to apply steering (strong vector)
MEASURE_LAYERS = [7, 14, 21, 28]  # Layers to measure similarity
ALPHA = 6.0               # Steering strength
STEER_DURATIONS = [1, 3, 5, 10]  # Number of initially-steered tokens
GENERATION_LENGTH = 30    # Total tokens to generate after steering stops
NUM_PROMPTS = 80          # Number of prompts to average over


def load_model_and_vectors():
    """Load model and pre-extracted steering vectors."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    sv_data = torch.load(STEERING_VECTOR_PATH, map_location="cpu", weights_only=True)
    steering_vectors = sv_data["steering_vectors"]
    print(f"Loaded steering vectors for layers: {list(steering_vectors.keys())}")

    return model, tokenizer, steering_vectors


def get_prompts(dataset_path, num_prompts):
    """Load and format prompts from CAA dataset."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Use items from the end of the dataset (not used for vector extraction)
    dataset = dataset[300:300 + num_prompts]
    prompts = []
    for item in dataset:
        question = item["question"]
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append({
            "prompt": prompt,
            "matching": item["answer_matching_behavior"].strip(),
            "not_matching": item["answer_not_matching_behavior"].strip(),
        })
    return prompts


def generate_with_hooks(model, tokenizer, input_ids, steering_vec, steer_layer,
                        alpha, steer_duration, total_length, measure_layers,
                        teacher_force_ids=None):
    """
    Generate tokens with steering for the first `steer_duration` tokens,
    then continue without steering.

    If teacher_force_ids is provided, use those tokens instead of sampling
    (but still don't apply steering after steer_duration).

    Returns:
        generated_ids: list of generated token ids
        similarities: dict mapping layer -> list of cosine similarities at each position
        projections: dict mapping layer -> list of signed projections at each position
    """
    device = input_ids.device
    hidden_dim = steering_vec.shape[0]

    # Normalize steering vector for cosine similarity measurement
    sv_normalized = {}
    for layer in measure_layers:
        # Load the steering vector for this measurement layer
        sv_data = torch.load(STEERING_VECTOR_PATH, map_location="cpu", weights_only=True)
        sv_layer = sv_data["steering_vectors"][layer].float().to(device)
        sv_normalized[layer] = sv_layer / sv_layer.norm()

    # Steering vector for injection
    sv_inject = steering_vec.to(device).to(model.dtype)

    similarities = {layer: [] for layer in measure_layers}
    projections = {layer: [] for layer in measure_layers}
    generated_ids = []

    current_ids = input_ids.clone()

    # Hook storage
    hook_handles = []
    captured_hidden_states = {}

    def make_capture_hook(layer_idx):
        """Capture hidden state at a specific layer."""
        def hook_fn(module, input, output):
            # For Qwen2, the output of each decoder layer is a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured_hidden_states[layer_idx] = hidden[:, -1, :].detach().float()
        return hook_fn

    def make_steer_hook(layer_idx):
        """Add steering vector at the specified layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # Only steer the last token position
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * sv_inject
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + alpha * sv_inject
                return output
        return hook_fn

    for step in range(total_length):
        captured_hidden_states.clear()
        for h in hook_handles:
            h.remove()
        hook_handles.clear()

        # Register capture hooks for measurement layers
        for layer_idx in measure_layers:
            handle = model.model.layers[layer_idx - 1].register_forward_hook(
                make_capture_hook(layer_idx)
            )
            hook_handles.append(handle)

        # Register steering hook if within steer_duration
        if step < steer_duration:
            handle = model.model.layers[steer_layer - 1].register_forward_hook(
                make_steer_hook(steer_layer)
            )
            hook_handles.append(handle)

        with torch.no_grad():
            outputs = model(current_ids)

        # Get next token
        if teacher_force_ids is not None and step < len(teacher_force_ids):
            next_token = teacher_force_ids[step].unsqueeze(0).unsqueeze(0)
        else:
            # Greedy decoding
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated_ids.append(next_token.item())

        # Measure similarities
        for layer_idx in measure_layers:
            if layer_idx in captured_hidden_states:
                h = captured_hidden_states[layer_idx].squeeze()
                h_normalized = h / h.norm()
                cos_sim = torch.dot(h_normalized, sv_normalized[layer_idx].to(h.device)).item()
                proj = torch.dot(h, sv_normalized[layer_idx].to(h.device)).item()
                similarities[layer_idx].append(cos_sim)
                projections[layer_idx].append(proj)
            else:
                similarities[layer_idx].append(0.0)
                projections[layer_idx].append(0.0)

        # Append token and continue
        current_ids = torch.cat([current_ids, next_token.to(device)], dim=1)

    # Clean up hooks
    for h in hook_handles:
        h.remove()

    return generated_ids, similarities, projections


def run_baseline_no_steering(model, tokenizer, input_ids, measure_layers, total_length):
    """Generate without any steering to establish baseline similarity."""
    device = input_ids.device

    sv_data = torch.load(STEERING_VECTOR_PATH, map_location="cpu", weights_only=True)
    sv_normalized = {}
    for layer in measure_layers:
        sv_layer = sv_data["steering_vectors"][layer].float().to(device)
        sv_normalized[layer] = sv_layer / sv_layer.norm()

    similarities = {layer: [] for layer in measure_layers}
    generated_ids = []
    current_ids = input_ids.clone()
    hook_handles = []
    captured_hidden_states = {}

    def make_capture_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured_hidden_states[layer_idx] = hidden[:, -1, :].detach().float()
        return hook_fn

    for step in range(total_length):
        captured_hidden_states.clear()
        for h in hook_handles:
            h.remove()
        hook_handles.clear()

        for layer_idx in measure_layers:
            handle = model.model.layers[layer_idx - 1].register_forward_hook(
                make_capture_hook(layer_idx)
            )
            hook_handles.append(handle)

        with torch.no_grad():
            outputs = model(current_ids)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        for layer_idx in measure_layers:
            if layer_idx in captured_hidden_states:
                h = captured_hidden_states[layer_idx].squeeze()
                h_normalized = h / h.norm()
                cos_sim = torch.dot(h_normalized, sv_normalized[layer_idx].to(h.device)).item()
                similarities[layer_idx].append(cos_sim)

        current_ids = torch.cat([current_ids, next_token.to(device)], dim=1)

    for h in hook_handles:
        h.remove()

    return generated_ids, similarities


def run_continuous_steering(model, tokenizer, input_ids, steering_vec, steer_layer,
                            alpha, measure_layers, total_length):
    """Generate with steering applied at every step (ceiling condition)."""
    return generate_with_hooks(
        model, tokenizer, input_ids, steering_vec, steer_layer,
        alpha, steer_duration=total_length, total_length=total_length,
        measure_layers=measure_layers
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model, tokenizer, steering_vectors = load_model_and_vectors()
    sv = steering_vectors[STEER_LAYER].float()
    print(f"Using steering vector at layer {STEER_LAYER}, norm={sv.norm().item():.4f}")

    prompts = get_prompts(DATASET_PATH, NUM_PROMPTS)
    print(f"Loaded {len(prompts)} prompts")

    total_gen = max(STEER_DURATIONS) + GENERATION_LENGTH

    # Storage for all results
    all_results = {
        "no_steering": {layer: [] for layer in MEASURE_LAYERS},
        "continuous": {layer: [] for layer in MEASURE_LAYERS},
    }
    for N in STEER_DURATIONS:
        all_results[f"initial_{N}_free"] = {layer: [] for layer in MEASURE_LAYERS}
        all_results[f"initial_{N}_teacher"] = {layer: [] for layer in MEASURE_LAYERS}

    generated_texts = {}  # Store some example generations

    for prompt_idx, prompt_data in enumerate(tqdm(prompts, desc="Running experiments")):
        prompt = prompt_data["prompt"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # 1. No steering baseline
        gen_ids, sims = run_baseline_no_steering(
            model, tokenizer, input_ids, MEASURE_LAYERS, total_gen
        )
        for layer in MEASURE_LAYERS:
            all_results["no_steering"][layer].append(sims[layer])
        if prompt_idx < 3:
            generated_texts[f"no_steering_{prompt_idx}"] = tokenizer.decode(gen_ids)

        # 2. Continuous steering (ceiling)
        gen_ids_cont, sims_cont, projs_cont = run_continuous_steering(
            model, tokenizer, input_ids, sv, STEER_LAYER,
            ALPHA, MEASURE_LAYERS, total_gen
        )
        for layer in MEASURE_LAYERS:
            all_results["continuous"][layer].append(sims_cont[layer])
        if prompt_idx < 3:
            generated_texts[f"continuous_{prompt_idx}"] = tokenizer.decode(gen_ids_cont)

        # 3. Initial-only steering for each duration N
        for N in STEER_DURATIONS:
            # Free autoregressive condition
            gen_ids_free, sims_free, projs_free = generate_with_hooks(
                model, tokenizer, input_ids, sv, STEER_LAYER,
                ALPHA, steer_duration=N, total_length=total_gen,
                measure_layers=MEASURE_LAYERS
            )
            for layer in MEASURE_LAYERS:
                all_results[f"initial_{N}_free"][layer].append(sims_free[layer])

            if prompt_idx < 3:
                generated_texts[f"initial_{N}_free_{prompt_idx}"] = tokenizer.decode(gen_ids_free)

            # Teacher-forced condition:
            # Use the tokens generated WITH continuous steering, but only apply
            # the steering vector for the first N tokens
            teacher_ids = torch.tensor(gen_ids_cont, device=DEVICE)
            gen_ids_tf, sims_tf, projs_tf = generate_with_hooks(
                model, tokenizer, input_ids, sv, STEER_LAYER,
                ALPHA, steer_duration=N, total_length=total_gen,
                measure_layers=MEASURE_LAYERS,
                teacher_force_ids=teacher_ids
            )
            for layer in MEASURE_LAYERS:
                all_results[f"initial_{N}_teacher"][layer].append(sims_tf[layer])

            if prompt_idx < 3:
                generated_texts[f"initial_{N}_teacher_{prompt_idx}"] = tokenizer.decode(gen_ids_tf)

    # Compute statistics
    results_stats = {}
    for condition, layer_data in all_results.items():
        results_stats[condition] = {}
        for layer, curves in layer_data.items():
            arr = np.array(curves)  # shape: (num_prompts, total_gen)
            results_stats[condition][layer] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "median": np.median(arr, axis=0).tolist(),
                "n": arr.shape[0],
            }

    # Save raw results
    save_data = {
        "config": {
            "model": MODEL_NAME,
            "steer_layer": STEER_LAYER,
            "measure_layers": MEASURE_LAYERS,
            "alpha": ALPHA,
            "steer_durations": STEER_DURATIONS,
            "generation_length": GENERATION_LENGTH,
            "num_prompts": NUM_PROMPTS,
            "total_gen": total_gen,
            "seed": SEED,
        },
        "results_stats": results_stats,
        "example_texts": generated_texts,
    }

    with open(os.path.join(RESULTS_DIR, "decay_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved results to {RESULTS_DIR}/decay_results.json")

    # Also save raw numpy arrays
    np.savez(
        os.path.join(RESULTS_DIR, "decay_raw.npz"),
        **{f"{cond}_layer{layer}": np.array(all_results[cond][layer])
           for cond in all_results for layer in MEASURE_LAYERS}
    )
    print(f"Saved raw arrays to {RESULTS_DIR}/decay_raw.npz")


if __name__ == "__main__":
    main()
