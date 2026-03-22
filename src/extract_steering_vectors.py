"""
Extract CAA steering vectors from contrastive pairs.

Uses sycophancy dataset from CAA repo. For each pair, runs both the
"matching behavior" and "not matching behavior" continuations through the model,
extracts residual stream activations at each layer, and computes the mean
difference vector (the steering vector).
"""

import json
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda:0"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_PAIRS = 300  # Number of contrastive pairs for extraction
DATASET_PATH = "code/CAA/datasets/generate/sycophancy/generate_dataset.json"
OUTPUT_DIR = "results/steering_vectors"


def format_prompt_pair(item):
    """Format a CAA item into two prompts: one with matching answer, one with not-matching."""
    question = item["question"]
    match = item["answer_matching_behavior"].strip()
    no_match = item["answer_not_matching_behavior"].strip()

    # Format as chat-style prompt (Qwen uses ChatML format)
    base = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    prompt_match = base + f"I would choose {match}"
    prompt_no_match = base + f"I would choose {no_match}"
    return prompt_match, prompt_no_match


def extract_activations(model, tokenizer, text, layers_to_extract):
    """Run a forward pass and extract residual stream activations at specified layers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers+1,) tensors of shape (batch, seq_len, hidden_dim)
    # Index 0 is embedding output, index i is output of layer i
    activations = {}
    for layer_idx in layers_to_extract:
        # Take the activation at the last token position
        act = outputs.hidden_states[layer_idx][0, -1, :].cpu().float()
        activations[layer_idx] = act
    return activations


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Extract at every 4th layer + first and last
    layers_to_extract = sorted(set([0, num_layers // 4, num_layers // 2,
                                     3 * num_layers // 4, num_layers]))
    print(f"Extracting at layers: {layers_to_extract}")

    # Load dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    # Use first NUM_PAIRS items
    dataset = dataset[:NUM_PAIRS]
    print(f"Using {len(dataset)} contrastive pairs")

    # Accumulate activation differences
    diffs = {layer: [] for layer in layers_to_extract}

    for item in tqdm(dataset, desc="Extracting steering vectors"):
        prompt_match, prompt_no_match = format_prompt_pair(item)

        act_match = extract_activations(model, tokenizer, prompt_match, layers_to_extract)
        act_no_match = extract_activations(model, tokenizer, prompt_no_match, layers_to_extract)

        for layer in layers_to_extract:
            diff = act_match[layer] - act_no_match[layer]
            diffs[layer].append(diff)

    # Compute mean steering vectors
    steering_vectors = {}
    for layer in layers_to_extract:
        stacked = torch.stack(diffs[layer])
        mean_vec = stacked.mean(dim=0)
        steering_vectors[layer] = mean_vec
        norm = mean_vec.norm().item()
        print(f"Layer {layer}: steering vector norm = {norm:.4f}")

    # Save
    save_path = os.path.join(OUTPUT_DIR, "sycophancy_steering_vectors.pt")
    torch.save({
        "steering_vectors": steering_vectors,
        "layers": layers_to_extract,
        "model_name": MODEL_NAME,
        "num_pairs": NUM_PAIRS,
        "num_layers": num_layers,
    }, save_path)
    print(f"Saved steering vectors to {save_path}")

    # Also save metadata
    meta = {
        "model_name": MODEL_NAME,
        "num_pairs": NUM_PAIRS,
        "layers_extracted": layers_to_extract,
        "num_model_layers": num_layers,
        "vector_norms": {str(l): steering_vectors[l].norm().item() for l in layers_to_extract},
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
