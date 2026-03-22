"""
Steering Vector Decay Experiment v2 - Delta-based measurement

Instead of measuring raw cosine similarity (which is tiny because steering
directions are orthogonal to most of the hidden state), we measure the
DIFFERENCE between steered and unsteered hidden states, projected onto the
steering direction. This directly captures the steering signal's persistence.

Approach:
1. Generate tokens with continuous steering (get token sequence T)
2. Teacher-force T without any steering → baseline hidden states
3. Teacher-force T with steering for first N tokens → steered hidden states
4. delta_t = steered_t - baseline_t; measure cos(delta_t, v) and ||delta_t||
5. Also: free generation with initial steering → divergent tokens, measure
   projection onto steering direction at each position

Additionally measures behavioral effect: probability shift toward sycophantic answer.
"""

import json
import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
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
MEASURE_LAYERS = [7, 14, 21, 28]
ALPHA = 8.0
STEER_DURATIONS = [1, 3, 5, 10]
TOTAL_GEN = 40
NUM_PROMPTS = 80


class SteeringHookManager:
    """Manages forward hooks for steering and activation capture."""

    def __init__(self, model, steering_vectors, steer_layer, alpha, measure_layers, device):
        self.model = model
        self.steering_vectors = steering_vectors
        self.steer_layer = steer_layer
        self.alpha = alpha
        self.measure_layers = measure_layers
        self.device = device
        self.hooks = []
        self.captured = {}
        self.do_steer = False

        # Pre-compute normalized steering vectors
        self.sv_normalized = {}
        for layer in measure_layers:
            sv = steering_vectors[layer].float().to(device)
            self.sv_normalized[layer] = sv / sv.norm()
        self.sv_inject = steering_vectors[steer_layer].to(device).to(model.dtype)

    def _capture_hook(self, layer_idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.captured[layer_idx] = h[:, -1, :].detach().float().clone()
        return hook_fn

    def _steer_hook(self):
        def hook_fn(module, input, output):
            if not self.do_steer:
                return output
            if isinstance(output, tuple):
                h = output[0]
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + self.alpha * self.sv_inject
                return (h,) + output[1:]
            else:
                output = output.clone()
                output[:, -1, :] = output[:, -1, :] + self.alpha * self.sv_inject
                return output
        return hook_fn

    def register(self):
        """Register all hooks."""
        self.remove()
        for layer_idx in self.measure_layers:
            h = self.model.model.layers[layer_idx - 1].register_forward_hook(
                self._capture_hook(layer_idx))
            self.hooks.append(h)
        # Steering hook
        h = self.model.model.layers[self.steer_layer - 1].register_forward_hook(
            self._steer_hook())
        self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def forward_step(self, input_ids, steer=False):
        """Run one forward pass, optionally with steering."""
        self.captured.clear()
        self.do_steer = steer
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs, dict(self.captured)


def teacher_force_with_recording(hook_mgr, input_ids, token_sequence, steer_duration):
    """
    Teacher-force a fixed token sequence, steering for first steer_duration tokens.
    Returns hidden states at each position for each measurement layer.
    """
    device = input_ids.device
    current_ids = input_ids.clone()
    hidden_states = {layer: [] for layer in hook_mgr.measure_layers}

    for step in range(len(token_sequence)):
        steer = (step < steer_duration)
        _, captured = hook_mgr.forward_step(current_ids, steer=steer)

        for layer in hook_mgr.measure_layers:
            hidden_states[layer].append(captured[layer].squeeze().cpu())

        next_token = torch.tensor([[token_sequence[step]]], device=device)
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return hidden_states


def free_generate_with_recording(hook_mgr, input_ids, steer_duration, total_length):
    """
    Generate freely, steering for first steer_duration tokens.
    Returns token ids and hidden states at each position.
    """
    device = input_ids.device
    current_ids = input_ids.clone()
    hidden_states = {layer: [] for layer in hook_mgr.measure_layers}
    generated_ids = []

    for step in range(total_length):
        steer = (step < steer_duration)
        outputs, captured = hook_mgr.forward_step(current_ids, steer=steer)

        for layer in hook_mgr.measure_layers:
            hidden_states[layer].append(captured[layer].squeeze().cpu())

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        current_ids = torch.cat([current_ids, next_token], dim=1)

    return generated_ids, hidden_states


def compute_delta_metrics(h_steered, h_baseline, sv_normalized):
    """
    Compute metrics from the difference between steered and baseline hidden states.

    Returns:
        delta_proj: Signed projection of delta onto steering direction
        delta_cos: Cosine similarity of delta with steering direction
        delta_norm: L2 norm of delta
    """
    delta = h_steered - h_baseline
    delta_norm = delta.norm().item()

    if delta_norm < 1e-8:
        return 0.0, 0.0, 0.0

    delta_proj = torch.dot(delta, sv_normalized).item()
    delta_cos = delta_proj / delta_norm

    return delta_proj, delta_cos, delta_norm


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

    hook_mgr = SteeringHookManager(
        model, steering_vectors, STEER_LAYER, ALPHA, MEASURE_LAYERS, DEVICE)
    hook_mgr.register()

    # Load prompts (from validation portion of dataset)
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
    dataset = dataset[300:300 + NUM_PROMPTS]

    # Results storage
    # For teacher-forced experiment: delta between steered and unsteered
    tf_delta_proj = {N: {L: [] for L in MEASURE_LAYERS} for N in STEER_DURATIONS}
    tf_delta_cos = {N: {L: [] for L in MEASURE_LAYERS} for N in STEER_DURATIONS}
    tf_delta_norm = {N: {L: [] for L in MEASURE_LAYERS} for N in STEER_DURATIONS}

    # Continuous steering deltas (ceiling)
    cont_delta_proj = {L: [] for L in MEASURE_LAYERS}

    # Free generation: raw projection onto steering direction
    free_proj = {N: {L: [] for L in MEASURE_LAYERS} for N in STEER_DURATIONS}
    no_steer_proj = {L: [] for L in MEASURE_LAYERS}
    cont_proj = {L: [] for L in MEASURE_LAYERS}

    # Behavioral: first-token answer
    behavioral = {"no_steer": [], "continuous": []}
    for N in STEER_DURATIONS:
        behavioral[f"initial_{N}"] = []

    example_texts = {}

    for idx, item in enumerate(tqdm(dataset, desc="Running v2 experiments")):
        question = item["question"]
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # --- Step 1: Generate with continuous steering (get reference token sequence) ---
        cont_ids, cont_h = free_generate_with_recording(
            hook_mgr, input_ids, steer_duration=TOTAL_GEN, total_length=TOTAL_GEN)

        for L in MEASURE_LAYERS:
            for t in range(TOTAL_GEN):
                p = torch.dot(cont_h[L][t].to(DEVICE), hook_mgr.sv_normalized[L]).item()
                if t < len(cont_proj[L]):
                    pass  # Will aggregate later
            cont_proj[L].append([
                torch.dot(cont_h[L][t].to(DEVICE), hook_mgr.sv_normalized[L]).item()
                for t in range(TOTAL_GEN)
            ])

        if idx < 3:
            example_texts[f"continuous_{idx}"] = tokenizer.decode(cont_ids)

        # Check behavioral: does first token contain matching answer?
        first_tok = tokenizer.decode([cont_ids[0]])
        behavioral["continuous"].append(
            item["answer_matching_behavior"].strip() in first_tok)

        # --- Step 2: Teacher-force the continuous-steered tokens WITHOUT any steering ---
        baseline_h = teacher_force_with_recording(
            hook_mgr, input_ids, cont_ids, steer_duration=0)

        for L in MEASURE_LAYERS:
            no_steer_proj[L].append([
                torch.dot(baseline_h[L][t].to(DEVICE), hook_mgr.sv_normalized[L]).item()
                for t in range(TOTAL_GEN)
            ])

        # --- Step 3: No-steering free generation ---
        no_steer_ids, _ = free_generate_with_recording(
            hook_mgr, input_ids, steer_duration=0, total_length=TOTAL_GEN)
        first_tok = tokenizer.decode([no_steer_ids[0]])
        behavioral["no_steer"].append(
            item["answer_matching_behavior"].strip() in first_tok)

        if idx < 3:
            example_texts[f"no_steer_{idx}"] = tokenizer.decode(no_steer_ids)

        # --- Step 4: For each N, teacher-force cont_ids with steering for first N ---
        for N in STEER_DURATIONS:
            steered_h = teacher_force_with_recording(
                hook_mgr, input_ids, cont_ids, steer_duration=N)

            for L in MEASURE_LAYERS:
                projs, coss, norms = [], [], []
                for t in range(TOTAL_GEN):
                    dp, dc, dn = compute_delta_metrics(
                        steered_h[L][t].to(DEVICE),
                        baseline_h[L][t].to(DEVICE),
                        hook_mgr.sv_normalized[L])
                    projs.append(dp)
                    coss.append(dc)
                    norms.append(dn)
                tf_delta_proj[N][L].append(projs)
                tf_delta_cos[N][L].append(coss)
                tf_delta_norm[N][L].append(norms)

        # --- Step 5: Continuous vs baseline delta (ceiling) ---
        for L in MEASURE_LAYERS:
            projs = []
            for t in range(TOTAL_GEN):
                dp, _, _ = compute_delta_metrics(
                    cont_h[L][t].to(DEVICE),
                    baseline_h[L][t].to(DEVICE),
                    hook_mgr.sv_normalized[L])
                projs.append(dp)
            cont_delta_proj[L].append(projs)

        # --- Step 6: Free generation with initial steering ---
        for N in STEER_DURATIONS:
            free_ids, free_h = free_generate_with_recording(
                hook_mgr, input_ids, steer_duration=N, total_length=TOTAL_GEN)

            for L in MEASURE_LAYERS:
                free_proj[N][L].append([
                    torch.dot(free_h[L][t].to(DEVICE), hook_mgr.sv_normalized[L]).item()
                    for t in range(TOTAL_GEN)
                ])

            first_tok = tokenizer.decode([free_ids[0]])
            behavioral[f"initial_{N}"].append(
                item["answer_matching_behavior"].strip() in first_tok)

            if idx < 3:
                example_texts[f"initial_{N}_free_{idx}"] = tokenizer.decode(free_ids)

    hook_mgr.remove()

    # Convert to arrays and save
    results = {
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
        "behavioral": {k: sum(v) / len(v) for k, v in behavioral.items()},
        "example_texts": example_texts,
    }

    # Save numpy arrays
    save_dict = {}
    for N in STEER_DURATIONS:
        for L in MEASURE_LAYERS:
            save_dict[f"tf_delta_proj_N{N}_L{L}"] = np.array(tf_delta_proj[N][L])
            save_dict[f"tf_delta_cos_N{N}_L{L}"] = np.array(tf_delta_cos[N][L])
            save_dict[f"tf_delta_norm_N{N}_L{L}"] = np.array(tf_delta_norm[N][L])
            save_dict[f"free_proj_N{N}_L{L}"] = np.array(free_proj[N][L])

    for L in MEASURE_LAYERS:
        save_dict[f"cont_delta_proj_L{L}"] = np.array(cont_delta_proj[L])
        save_dict[f"no_steer_proj_L{L}"] = np.array(no_steer_proj[L])
        save_dict[f"cont_proj_L{L}"] = np.array(cont_proj[L])

    np.savez(os.path.join(RESULTS_DIR, "decay_v2_raw.npz"), **save_dict)

    with open(os.path.join(RESULTS_DIR, "decay_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== BEHAVIORAL RESULTS ===")
    for k, v in results["behavioral"].items():
        print(f"  {k}: {v:.1%} sycophantic")

    print(f"\nSaved results to {RESULTS_DIR}/decay_v2_*.json/npz")


if __name__ == "__main__":
    main()
