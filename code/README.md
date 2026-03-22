# Cloned Repositories for Steering Vector Research

This directory contains shallow clones (`--depth 1`) of key repositories for the
steering vector decay research project. Each entry describes the repository's
purpose, authors, and relevance to the project.

---

## 1. CAA — Contrastive Activation Addition

**Directory:** `CAA/`
**Source:** https://github.com/nrimsky/CAA
**Authors:** Nina Rimsky et al.
**Paper:** *Steering Llama 2 via Contrastive Activation Addition* (arXiv:2312.06681)

### What it provides
The original implementation of Contrastive Activation Addition (CAA) applied to
Llama 2. Steering vectors are computed as the mean difference between residual
stream activations on positive and negative prompt pairs, then injected at
inference time. Covers behaviors such as corrigibility, sycophancy, hallucination,
survival instinct, and myopia. Includes datasets of contrastive prompt pairs drawn
from Anthropic model-written evals and GPT-4-generated data (50 test pairs per
behavior, hundreds of generation pairs), plus evaluation scripts using GPT-4 as a
judge and TruthfulQA / MMLU benchmarks.

### Relevance
Core baseline for CAA-style steering. The vector generation pipeline, layer-wise
injection approach, and behavioral datasets are directly reusable.

---

## 2. activation_additions — Algebraic Value Editing (ActAdd)

**Directory:** `activation_additions/`
**Source:** https://github.com/montemac/activation_additions
**Authors:** Monte MacDiarmid, Alex Turner, et al.
**Paper:** *Steering Language Models With Activation Engineering* (arXiv:2308.10248)

### What it provides
The original "X-vector" / ActAdd codebase. Steering vectors are constructed by
running a prompt (e.g. "Love") through a forward pass and recording the residual
stream at a chosen layer, then adding a scaled version to future forward passes at
the same hook point. "X-vectors" are difference vectors between two contrastive
prompts. Uses TransformerLens hooks on GPT-2 family models. Includes example
scripts for basic functionality and padding strategies for mismatched token lengths.

### Relevance
Establishes the algebraic framing of activation addition and the "coefficient"
parametrization. Important comparison point for single-prompt vs. contrastive
vector construction.

---

## 3. steering-bench — Generalization and Reliability of Steering Vectors

**Directory:** `steering-bench/`
**Source:** https://github.com/dtch1997/steering-bench
**Authors:** Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige,
Adria Garriga-Alonso, Robert Kirk (FAR AI / UCL)
**Paper:** *Analyzing the Generalization and Reliability of Steering Vectors*
(arXiv:2407.12404)

### What it provides
A benchmark and evaluation suite for steering vectors. Provides:
- Layer sweep experiments (extract and apply steering vectors across all layers to
  select optimal layers by steerability).
- Steering generalization experiments (vary user/system prompts to test robustness).
- Off-the-shelf components: `Pipeline` (steered model wrapper), `Formatter` (prompt
  scaffolding), `PipelineHook` / `SteeringHook` (intervention abstractions),
  steerability metrics.
- Built on top of the `steering-vectors` library (see below).

### Relevance
Directly addresses questions of how steering vectors generalize across prompts and
layers — central to understanding decay and transfer.

---

## 4. tense-aspect — Grammatical Tense/Aspect Steering

**Directory:** `tense-aspect/`
**Source:** https://github.com/klerings/tense-aspect
**Authors:** Klerings et al.
**Paper:** *Steering Language Models in Multi-Token Generation: A Case Study on
Tense and Aspect* (EMNLP 2025)

### What it provides
Investigates how LLMs internally encode grammatical tense and aspect using linear
discriminant analysis (LDA) and contrastive steering vectors. Covers:
- Extraction of hidden states across layers for PropBank-labeled and synthetic data.
- Training of probing classifiers per tense/aspect category.
- LDA-based feature direction extraction (builds on KihoPark/
  LLM_Categorical_Hierarchical_Representations).
- Multiple steering methods: `decreasing_intensity`, `every_second_step`,
  `subtract_and_add`, `subtract_proj`, `first_step_only`, and standard greedy/
  non-greedy variants.
- Evaluation on BIG-bench tense subset.
- Models: Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct and others via config files.

### Relevance
Directly relevant to steering vector decay: the `decreasing_intensity` method is
an explicit implementation of decaying steering strength across generation steps.
Provides a working multi-token steering framework and empirical findings on
side-effect minimization.

---

## 5. TransformerLens

**Directory:** `TransformerLens/`
**Source:** https://github.com/TransformerLensOrg/TransformerLens
**Authors:** Originally Neel Nanda; maintained by Bryce Meyer and the
TransformerLensOrg community.

### What it provides
A mechanistic interpretability library for GPT-2-style language models. Supports
50+ open-source models. Key features:
- `HookedTransformer`: loads pretrained models and exposes all internal activations
  via hook points.
- Activation caching (`run_with_cache`).
- Hook functions for editing, removing, or replacing activations at any layer
  during a forward pass.
- Used as the underlying infrastructure in both `activation_additions` and many
  other steering vector projects.

### Relevance
Primary tooling for hooking into model internals to inject, modify, or monitor
steering vectors. Essential for implementing custom decay schedules.

---

## 6. angular-steering

**Directory:** `angular-steering/`
**Source:** https://github.com/lone17/angular-steering
**Authors:** lone17 et al.

### What it provides
Implements "Angular Steering" — steering vectors defined on a plane (not a single
direction) with a target angle. Key components:
- `angular_steering.ipynb`: activation analysis, direction extraction, plane
  construction.
- `generate_responses.py`: batch steered generation via a custom vLLM fork.
- `vllm_angular_steering.py`: newer vLLM v1-native implementation (no fork needed).
- Evaluation scripts: substring matching, LlamaGuard 3, HarmBench, LLM-as-a-judge.
- Perplexity evaluation and tinyBenchmarks integration.
- Supports random plane generation for ablation.
- Gradio chat UI and OpenAI-compatible endpoint server.

### Relevance
Angular representation of steering vectors is a geometrically principled
alternative to scalar-multiplied directions. The plane-based formulation may
connect to decay dynamics in angle space.

---

## 7. SAELens — Sparse Autoencoder Training and Analysis

**Directory:** `SAELens/`
**Source:** https://github.com/jbloomaus/SAELens
**Authors:** Joseph Bloom, Curt Tigges, Anthony Duong, David Chanin
(decoderesearch / Decode Research)

### What it provides
A library for training and analyzing Sparse Autoencoders (SAEs) on language
models, aimed at mechanistic interpretability. Features:
- SAE training on any PyTorch model (not TransformerLens-only).
- Deep TransformerLens integration via `HookedSAETransformer`.
- Compatibility with Hugging Face Transformers and NNsight via `encode()` /
  `decode()` methods.
- Pre-trained SAE loading from Neuronpedia.
- Feature visualization with SAE-Vis.
- Tutorials covering training, feature analysis, and logit lens.

### Relevance
SAE-decomposed residual streams can be used to understand which features are
activated/suppressed by steering vectors, and to construct interpretable steering
directions in feature space rather than raw activation space.

---

## 8. steering-vectors — Python Library for Steering Vectors

**Directory:** `steering-vectors/`
**Source:** https://github.com/steering-vectors/steering-vectors
**Authors:** steering-vectors org (David Chanin, Daniel Tan, et al.)

### What it provides
A general-purpose Python library (`pip install steering-vectors`) for training and
applying steering vectors to Hugging Face transformer models. Supports GPT,
LLaMA, Gemma, Mistral, Pythia, and more. Inspired by and compatible with both
Rimsky et al. (CAA) and Zou et al. (RepE). Provides clean abstractions for:
- Training steering vectors from contrastive prompt pairs.
- Applying vectors at inference time with configurable hook points.
- Integration with `steering-bench` for evaluation.

### Relevance
The cleanest reusable implementation for standard CAA-style steering. Good
starting point for implementing decay variants without rebuilding from scratch.

---

## 9. representation-engineering — RepE

**Directory:** `representation-engineering/`
**Source:** https://github.com/andyzoujm/representation-engineering
**Authors:** Andy Zou, Long Phan, Sarah Chen, James Campbell, et al. (Zou lab /
CMU / UIUC / Berkeley)
**Paper:** *Representation Engineering: A Top-Down Approach to AI Transparency*
(arXiv:2310.01405)

### What it provides
The official RepE codebase. Uses population-level representations (from positive/
negative prompt pairs) extracted via PCA or linear probing to build "reading
vectors" and "control vectors". Integrates with Hugging Face pipelines:
- `rep-reading` pipeline: classify hidden states along a concept direction.
- `rep-control` pipeline: inject control vectors to steer model behavior.
Covers safety-relevant applications: truthfulness, memorization, power-seeking,
emotion, and more.

### Relevance
RepE is an alternative construction to CAA (PCA-based rather than mean-difference).
Understanding the relationship between RepE and CAA vectors, and whether decay
behavior differs between them, is a natural research question for this project.

---

## Summary Table

| Directory                  | Key Method            | Model Family          | Primary Use                          |
|----------------------------|-----------------------|-----------------------|--------------------------------------|
| `CAA/`                     | Mean-diff (contrastive) | Llama 2             | Behavioral steering, datasets        |
| `activation_additions/`    | X-vector (single/diff)  | GPT-2               | Algebraic framing, coefficient study |
| `steering-bench/`          | CAA via steering-vectors | Llama family       | Evaluation, generalization analysis  |
| `tense-aspect/`            | LDA + contrastive SV    | Qwen2.5, LLaMA-3    | Multi-token decay, grammar control   |
| `TransformerLens/`         | Hook infrastructure     | GPT-2 family, 50+   | Model internals, hooking             |
| `angular-steering/`        | Angular/plane steering  | LLaMA family        | Geometric steering, jailbreak eval   |
| `SAELens/`                 | SAE decomposition       | GPT, LLaMA, etc.    | Feature-space steering, interp       |
| `steering-vectors/`        | CAA (library)           | All HF models       | General-purpose steering library     |
| `representation-engineering/` | PCA/linear probe (RepE) | LLaMA family     | Population-level representation ctrl |

---

*All repositories cloned with `--depth 1` on 2026-03-22.*
