# Resources Catalog

## Summary
This document catalogs all resources gathered for the Steering Vector Decay research project.
The project investigates how steering vector influence decays over subsequent generated tokens
in language models, and whether the decay differs under teacher-forcing vs. continuous steering.

## Papers
Total papers downloaded: 15

| Title | Authors | Year | File | Key Relevance |
|-------|---------|------|------|---------------|
| Activation Engineering (ActAdd) | Turner et al. | 2023 | papers/2308.10248_*.pdf | Foundational steering method |
| Contrastive Activation Addition (CAA) | Rimsky et al. | 2024 | papers/2312.06681_*.pdf | Standard CAA + behavioral datasets |
| Dynamic Activation Composition | Scalena et al. | 2024 | papers/2406.17563_*.pdf | **Start/Fixed/Dim/Dyn comparison** |
| SAE vs. Activation Difference | Xie | 2025 | papers/2510.01246_*.pdf | **Token-wise decay formula** |
| Multi-Token Steering (Tense/Aspect) | Klerings et al. | 2025 | papers/2509.12065_*.pdf | **Duration/location effects** |
| Inference-Time Intervention (ITI) | Li et al. | 2023 | papers/2306.03341_*.pdf | Truthfulness steering baseline |
| KV Cache Steering | Belitsky et al. | 2025 | papers/2507.08799_*.pdf | **One-shot vs continuous** |
| Generalization & Reliability | Tan et al. | 2024 | papers/2407.12404_*.pdf | Steering failure modes |
| Refusal Single Direction | Arditi et al. | 2024 | papers/2406.11717_*.pdf | Persistent 1D feature |
| Latent Steering Vectors | Subramani et al. | 2022 | papers/2205.05124_*.pdf | Early steering work |
| Representation Engineering | Zou et al. | 2023 | papers/2310.01405_*.pdf | RepE framework |
| Mean-Centring | Jorgensen et al. | 2023 | papers/2312.03813_*.pdf | Improved SV computation |
| Steering Vector Fields | Li et al. | 2026 | papers/2602.01654_*.pdf | Context-dependent steering |
| Mechanistic Indicators | Jafari et al. | 2026 | papers/2602.01716_*.pdf | Effectiveness across steps |
| PIXEL Adaptive Steering | Yu et al. | 2025 | papers/2510.10205_*.pdf | Position-wise adaptation |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA | HuggingFace | 817 examples | Truthfulness MC | datasets/truthful_qa/ | Standard truthfulness benchmark |
| Alpaca (subset) | HuggingFace | 1000 examples | General QA | datasets/alpaca_subset/ | Language steering evaluation |
| BeaverTails (subset) | HuggingFace | 500 examples | Safety | datasets/beavertails_subset/ | Safety steering evaluation |
| CAA Behavioral | Cloned repo | 7 categories | Behavior MC | code/CAA/datasets/ | Sycophancy, refusal, hallucination, etc. |

See datasets/README.md for download instructions.

## Code Repositories
Total repositories cloned: 9

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| CAA | github.com/nrimsky/CAA | Original CAA implementation + datasets | code/CAA/ |
| activation_additions | github.com/montemac/activation_additions | ActAdd/X-vector implementation | code/activation_additions/ |
| steering-bench | github.com/dtch1997/steering-bench | Steering vector benchmark | code/steering-bench/ |
| tense-aspect | github.com/klerings/tense-aspect | Multi-token steering + decay methods | code/tense-aspect/ |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability hooks | code/TransformerLens/ |
| angular-steering | github.com/lone17/angular-steering | Rotation-based steering | code/angular-steering/ |
| SAELens | github.com/jbloomaus/SAELens | SAE training & analysis | code/SAELens/ |
| steering-vectors | github.com/steering-vectors/steering-vectors | General-purpose CAA library | code/steering-vectors/ |
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE: PCA/probe-based control | code/representation-engineering/ |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with "activation steering vectors language models decay" in diligent mode (121 results)
2. Filtered by relevance score >= 2 (yielded ~40 papers)
3. Identified 15 most relevant papers for download, prioritizing:
   - Papers directly addressing steering duration/decay/schedules
   - Foundational steering methods (ActAdd, CAA, ITI, RepE)
   - Papers studying steering failure modes and reliability

### Selection Criteria
- Direct relevance to steering vector decay hypothesis
- Empirical evidence about steering persistence over tokens
- Methodological contributions (decay formulas, adaptive strategies)
- High citation count for foundational work

### Key Insights from Literature
1. **Steering decays when applied only initially**: Scalena et al. (2024) show Start steering fails for most properties
2. **Constant steering causes artifacts**: Both Xie (2025) and Scalena et al. find repetition/degeneration with constant high-intensity steering
3. **Optimal steering intensity decreases over time**: Scalena et al.'s Dynamic method shows α drops sharply after first few tokens
4. **Decay is property-dependent**: Language steering may need less ongoing steering than safety or formality
5. **KV cache provides persistent steering**: Belitsky et al. show one-shot KV modification persists, unlike activation modifications

## Recommendations for Experiment Design

1. **Primary dataset(s)**: CAA behavioral datasets (clean A/B format for measuring steering effect per position) + TruthfulQA
2. **Baseline methods**: Constant CAA, no steering, linear diminishing, Xie decay formula
3. **Evaluation metrics**: Per-token steering accuracy, cosine similarity to steering direction, perplexity
4. **Code to adapt/reuse**:
   - `steering-vectors` library for clean CAA implementation
   - `TransformerLens` for hook-based activation extraction
   - `tense-aspect` repo includes a `decreasing_intensity` method
   - `CAA` repo for behavioral datasets and baseline implementation
5. **Experimental approach**:
   - Apply steering at steps 0..N, then stop steering and continue generation
   - At each subsequent step, measure projection of activations onto steering direction
   - Compare: (a) teacher-forced tokens from steered generation without further steering, (b) fully autoregressive with steering only at initial steps, (c) continuous steering throughout
   - Measure behavioral outcome (e.g., does the model still choose the steered answer?)
