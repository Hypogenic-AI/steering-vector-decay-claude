# Research Plan: Steering Vector Decay

## Motivation & Novelty Assessment

### Why This Research Matters
Activation steering is one of the simplest and most effective inference-time methods for controlling LLM behavior. Practitioners must decide whether to steer continuously (risking degeneration) or for a limited window. Understanding the decay dynamics of steering influence directly informs these engineering decisions—yet no systematic study exists of the actual decay curves.

### Gap in Existing Work
Scalena et al. (2024) show that "start-only" steering fails and dynamic alpha drops sharply after initial tokens, but they don't isolate the decay mechanism. Xie (2025) proposes a decay formula but doesn't measure the natural decay. Nobody has compared how teacher-forcing steered tokens (without continued steering) affects the persistence of the steering signal in hidden states.

### Our Novel Contribution
We systematically measure the decay of steering vector influence in hidden states as a function of:
1. Number of initially-steered tokens (N = 1, 3, 5, 10)
2. Post-steering regime: (a) free autoregressive generation, (b) teacher-forced replay of steered tokens without the vector
3. Layer of intervention

This provides the first empirical decay curves and isolates the autoregressive feedback mechanism.

### Experiment Justification
- **Experiment 1 (Steering Vector Extraction)**: Needed to obtain reliable CAA vectors for sycophancy.
- **Experiment 2 (Decay Curves)**: Core experiment—apply steering for N tokens, then measure cosine similarity of hidden states to the steering direction at each subsequent position.
- **Experiment 3 (Teacher-Forcing Comparison)**: Tests whether the steered tokens themselves carry the signal (via autoregressive context) or whether it must be injected into hidden states.
- **Experiment 4 (Layer Dependence)**: Checks whether decay varies by layer.

## Research Question
How quickly does the influence of an activation steering vector decay in hidden states after it is no longer applied, and does teacher-forcing steered tokens (without continued steering) change this decay?

## Hypothesis Decomposition
1. **H1**: Cosine similarity between hidden states and the steering direction decreases monotonically after steering stops.
2. **H2**: The decay rate depends on how many tokens were steered (more initial steering = slower decay).
3. **H3**: Teacher-forcing steered tokens without the vector produces slower decay than free generation, because the steered tokens themselves carry behavioral information in the autoregressive context.
4. **H4**: Decay is faster in later layers (closer to output) than earlier layers.

## Proposed Methodology

### Approach
Use Contrastive Activation Addition (CAA) on a small instruction-tuned model (Gemma-2-2B-IT). Extract sycophancy steering vectors from the CAA behavioral dataset. Perform controlled generation experiments with hooks to inject and measure activations.

### Model Choice
- **Primary**: `google/gemma-2-2b-it` — small enough for fast iteration, large enough for meaningful steering effects, well-supported in HuggingFace
- **Justification**: Used by Xie (2025) for similar decay work. 2B params fits easily on one A6000.

### Experimental Steps

1. **Extract steering vectors**: Run 500 sycophancy contrastive pairs through the model, extract residual stream activations at each layer, compute mean difference vectors.

2. **Validate steering works**: Confirm that continuous steering shifts model behavior on held-out sycophancy questions.

3. **Decay curve experiment**:
   - For N ∈ {1, 3, 5, 10}: apply steering vector (α=4.0) for first N generation tokens
   - Generate 30 additional tokens without steering
   - At each token position, record cosine similarity of residual stream to steering direction
   - Repeat for 100 prompts, average curves

4. **Teacher-forcing experiment**:
   - For each prompt, first generate 30 tokens WITH continuous steering (save the token sequence)
   - Then re-run: teacher-force those same tokens but only apply the steering vector for the first N tokens
   - Compare decay curves between conditions

5. **Layer analysis**: Repeat core experiment measuring at layers {early, middle, late}.

### Baselines
- **No steering**: Measure baseline cosine similarity (noise floor)
- **Continuous steering**: Measure similarity with vector always applied (ceiling)
- **Free generation (initial-only)**: Apply for N tokens, then free generation
- **Teacher-forced (initial-only)**: Apply for N tokens, then teacher-force without vector

### Evaluation Metrics
1. **Cosine similarity to steering direction** at each token position (primary)
2. **Projection magnitude** (signed projection onto steering direction)
3. **Behavioral outcome**: Whether model still produces "matching behavior" answers
4. **Half-life**: Tokens until cosine similarity drops to 50% of steered value

### Statistical Analysis Plan
- Report mean ± std across prompts for each condition
- Use paired t-tests or Wilcoxon signed-rank tests comparing teacher-forced vs. free generation decay
- Fit exponential decay curves: sim(t) = A * exp(-t/τ) + C
- Report τ (time constant) and goodness-of-fit (R²)

## Expected Outcomes
- H1: Expect monotonic but potentially non-exponential decay
- H2: More initial steering tokens → higher initial similarity → but similar decay rate
- H3: Teacher-forcing should produce slower decay (the steered tokens are "in-distribution" for sycophantic behavior)
- H4: Later layers may show faster decay as they're more task-specific

## Timeline
- Phase 1 (Planning): 15 min ✓
- Phase 2 (Setup + Extraction): 20 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- Model may not show clean sycophancy steering with Gemma-2-2B → fallback to larger model or different behavior
- Cosine similarity may be noisy at single-token level → average over many prompts
- Teacher-forcing changes the computational graph → need careful hook placement

## Success Criteria
- Clear decay curves with measurable time constants
- Statistically significant difference between teacher-forced and free generation conditions
- Results consistent across multiple prompts and layers
