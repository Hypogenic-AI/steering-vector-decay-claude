# Literature Review: Steering Vector Decay

## Research Area Overview

Activation steering (also called representation engineering) is a family of inference-time techniques that modify a language model's internal representations to control its behavior without fine-tuning. A steering vector—typically computed as the mean difference in activations between contrastive prompt pairs—is added to the model's hidden states during the forward pass. While these methods have shown effectiveness for controlling properties like sentiment, safety, truthfulness, and language, a critical open question is **how the influence of a steering vector decays over subsequent generated tokens**, particularly when the vector is applied only during early generation steps.

This review synthesizes the key literature relevant to the hypothesis that steering vector influence decays over tokens, and that the rate and nature of this decay depends on whether generated tokens are teacher-forced (fed back without further steering) versus continuously steered.

---

## Key Papers

### 1. Turner et al. (2023) - "Steering Language Models With Activation Engineering"
- **arXiv**: 2308.10248
- **Key Contribution**: Introduces Activation Addition (ActAdd), the foundational method for inference-time steering via contrastive activation differences.
- **Methodology**: Computes a steering vector from the difference in activations on a pair of contrastive prompts (e.g., "Love" vs. "Hate"). Adds this vector to the residual stream at a chosen layer during the forward pass.
- **Relevance to Decay**: ActAdd applies the steering vector at specific token positions. The paper demonstrates that a single-position intervention can influence subsequent generation, suggesting some persistence of the steering signal through the autoregressive process. However, it does not systematically study how this influence attenuates over tokens.

### 2. Rimsky et al. (2024) - "Steering Llama 2 via Contrastive Activation Addition"
- **arXiv**: 2312.06681 | **Citations**: 579
- **Key Contribution**: Introduces Contrastive Activation Addition (CAA), computing steering vectors by averaging activation differences across many contrastive pairs. Applies vectors at all token positions after the user prompt.
- **Methodology**: Steering vectors are added at every generated token position with a fixed coefficient. Evaluated on multiple-choice behavioral questions (sycophancy, hallucination, refusal, etc.) and open-ended generation.
- **Relevance to Decay**: CAA applies steering continuously at every step, which masks any natural decay. The paper does not compare continuous vs. initial-only steering, but the decision to steer at every position implicitly suggests that single-step steering may be insufficient.
- **Datasets**: Behavioral question datasets in A/B multiple-choice format (7 categories). Available in the cloned repo.

### 3. Scalena et al. (2024) - "Multi-property Steering of LLMs with Dynamic Activation Composition"
- **arXiv**: 2406.17563 | **Citations**: 26
- **Key Contribution**: Directly compares multiple steering schedules: Start (first step only), Fixed (constant), Diminishing (linear decay), and Dynamic (KL-based adaptive). Proposes Dynamic Activation Composition.
- **Methodology**: Extracts per-step steering vectors from contrastive ICL pairs. Tests on language, safety, and formality properties using Mistral 7B.
- **Critical Findings for Decay**:
  - **Start (first-step-only) steering fails** to maintain conditioning across generation for most properties. The first generated token may be in the correct language, but subsequent tokens revert to English.
  - **Fixed steering works but degrades fluency** at high intensities, causing repetition and nonsensical output.
  - **Diminishing steering** preserves effectiveness while reducing perplexity, but optimal parameters are property-dependent.
  - **Dynamic steering** (their proposed method) shows that **α_i generally decreases sharply after the first few generated tokens** (Figure 5), providing direct evidence that most steering work happens early.
  - The paper demonstrates that "previously generated property-aligned tokens become increasingly influential as generation progresses," meaning the model's own autoregressive context gradually takes over from the external steering signal.
- **Datasets Used**: Alpaca (language), BeaverTails (safety), GYAFC/XFORMAL (formality).

### 4. Xie (2025) - "A Comparative Analysis of SAE and Activation Difference in Language Model Steering"
- **arXiv**: 2510.01246
- **Key Contribution**: Proposes a **token-wise decaying steering strategy** and directly identifies that constant steering produces degenerate outputs (word repetition).
- **Methodology**: Uses SAE features for steering with a decay formula: α_t = α · (1/(1+ω·t))^k. Tests on Gemma-2-2B and Gemma-2-9B for math reasoning (GSM8K, SVAMP, MAWPS, ASDIV) and instruction following (IFEval).
- **Critical Findings for Decay**:
  - Constant SAE steering at high strength produces "Since Since Since..." repetition.
  - The decaying strategy applies strong steering at the beginning to initiate the desired behavior (e.g., chain-of-thought reasoning), then reduces it to let the model's learned priors take over.
  - With k=3-4 and appropriate ω, the coefficient decays to ~1 after about 8-9 steps.
  - L2-norm normalization of steered activations helps maintain representational stability.
  - SAE steering with decay behaves similarly to appending a guiding token to the prompt.
- **Datasets Used**: GSM8K, SVAMP, MAWPS, ASDIV (math reasoning), IFEval (instruction following).

### 5. Klerings et al. (2025) - "Steering Language Models in Multi-Token Generation"
- **arXiv**: 2509.12065
- **Key Contribution**: Systematically studies steering strength, location, and **duration** as factors in multi-token generation, using grammatical tense/aspect as the steering target.
- **Methodology**: Uses LDA-based feature directions to steer tense and aspect in Llama-3.1-8B and Qwen-2.5-7B. Compares prompt-based vs. generation-time steering, single-token vs. multi-token interventions.
- **Critical Findings for Decay**:
  - **Generation-time steering is more effective** than prompt-based steering—the effect of steering during the prompt decays by the time generation begins.
  - **Late or extended steering causes topic shift and degeneration**—over-steering after the behavior is established is harmful.
  - **Steering just before the verb works best**—timing matters more than duration.
  - Activation norms increase across layers, requiring proportionally larger α values in deeper layers.
  - The same steering vector can produce success, no effect, or degeneration depending on where and for how long it is applied.
- **Code**: https://github.com/klerings/tense-aspect

### 6. Li et al. (2023) - "Inference-Time Intervention: Eliciting Truthful Answers"
- **arXiv**: 2306.03341 | **Citations**: 951
- **Key Contribution**: Introduces ITI, which shifts activations across a limited number of attention heads to improve truthfulness.
- **Relevance to Decay**: ITI applies continuous intervention at every forward pass. The high citation count and widespread adoption suggest that continuous steering is the default, implying researchers assume single-step steering would be insufficient.

### 7. Belitsky et al. (2025) - "KV Cache Steering for Controlling Frozen LLMs"
- **arXiv**: 2507.08799
- **Key Contribution**: Proposes **one-shot** steering via KV cache modification, contrasting with continuous activation steering.
- **Methodology**: Modifies the KV cache once after prompt processing. Because cached representations are fixed, they implicitly influence all future tokens without repeated intervention.
- **Critical Findings for Decay**:
  - One-shot KV cache steering is effective because the modified keys/values persist in the attention mechanism, providing an implicit "memory" of the steering direction.
  - This contrasts with activation steering, where a one-shot intervention to the hidden state at step 0 would be diluted by subsequent layer operations and autoregressive generation.
  - Cache steering is more stable and less sensitive to hyperparameters than continuous activation steering.
  - This paper implicitly demonstrates that **activation steering decays** (requiring continuous intervention) while **KV cache modifications persist** (one-shot is sufficient).

### 8. Tan et al. (2024) - "Analyzing the Generalization and Reliability of Steering Vectors"
- **arXiv**: 2407.12404 | **Citations**: 69
- **Key Contribution**: Systematically evaluates generalization and reliability of steering vectors, showing substantial limitations.
- **Findings**: Steerability is highly variable across inputs. Spurious biases contribute to effectiveness. Steering vectors are brittle to reasonable prompt changes. Overall, steering can work but has technical difficulties at scale.
- **Code**: https://github.com/dtch1997/steering-bench

### 9. Arditi et al. (2024) - "Refusal in Language Models Is Mediated by a Single Direction"
- **arXiv**: 2406.11717 | **Citations**: 519
- **Key Contribution**: Shows refusal is mediated by a one-dimensional subspace in the residual stream. Erasing this direction prevents refusal; adding it elicits refusal.
- **Relevance**: Demonstrates that some behaviors have very strong, persistent linear representations. For such behaviors, steering may decay more slowly because the direction is deeply embedded in the model's processing.

### 10. Subramani et al. (2022) - "Extracting Latent Steering Vectors from Pretrained Language Models"
- **arXiv**: 2205.05124 | **Citations**: 167
- **Key Contribution**: Early work showing steering vectors exist in pretrained LMs. Finds that adding vectors to hidden states generates target sentences with >99 BLEU.
- **Relevance**: Demonstrates that the steering effect can be very precise when applied correctly, suggesting the decay question is about the autoregressive feedback loop rather than the vector's inherent quality.

---

## Common Methodologies

### Steering Vector Extraction
- **Mean Activation Difference (MeanActDiff/CAA)**: Average difference between activations on positive vs. negative contrastive pairs. Used by Turner et al., Rimsky et al., Scalena et al.
- **SAE Feature Directions**: Use sparse autoencoder decoder columns as interpretable steering directions. Used by Xie (2025).
- **LDA-based Directions**: Linear discriminant analysis to find categorical feature directions. Used by Klerings et al.
- **Optimized Vectors**: Gradient descent on a single example. Used by Dunefsky & Cohan (2025).

### Steering Application Strategies
1. **Constant (Fixed)**: Same α at every step. Standard approach (CAA, ITI).
2. **Initial-only (Start)**: Apply only at first step. Shown to be insufficient by Scalena et al.
3. **Diminishing (Dim)**: Linear or polynomial decay. Used by Scalena et al., Xie (2025).
4. **Dynamic (Dyn)**: KL-divergence-based adaptive intensity. Proposed by Scalena et al.
5. **One-shot KV Cache**: Modify KV cache once before generation. Proposed by Belitsky et al.

### Evaluation Metrics
- **Conditioning Accuracy**: Property-specific classifiers, language detection, LLM-as-judge
- **Fluency**: Perplexity (∆PPL relative to ICL baseline)
- **Side Effects**: BERTScore (topic shift), repetition detection, degeneration rate
- **Task Performance**: Accuracy on downstream benchmarks (TruthfulQA, GSM8K, etc.)

---

## Datasets in the Literature

| Dataset | Used By | Task | Available |
|---------|---------|------|-----------|
| TruthfulQA | Li et al. 2023 (ITI) | Truthfulness MC | ✓ Downloaded |
| Alpaca | Scalena et al. 2024 | Language steering | ✓ Downloaded |
| BeaverTails | Scalena et al. 2024 | Safety steering | ✓ Downloaded |
| CAA Behavioral | Rimsky et al. 2024 | Sycophancy, refusal, etc. | ✓ In code/CAA/ |
| GSM8K | Xie 2025 | Math reasoning | Via HuggingFace |
| IFEval | Xie 2025 | Instruction following | Via HuggingFace |
| Penn Treebank + BIG-bench | Klerings et al. 2025 | Tense/aspect | Via HuggingFace |
| GYAFC / XFORMAL | Scalena et al. 2024 | Formality | Via HuggingFace |

---

## Gaps and Opportunities

1. **No systematic study of decay curves**: While Scalena et al. and Xie show that steering needs to decay, nobody has systematically measured the decay curve of steering influence over tokens. The proposed research would directly fill this gap.

2. **Teacher-forcing vs. autoregressive feedback**: The hypothesis distinguishes between feeding back steered tokens with vs. without continued steering. Scalena et al.'s Dyn method implicitly adapts to this (α drops when the prefix is already conditioned), but the mechanism hasn't been isolated.

3. **Property-dependent decay rates**: Scalena et al. show different properties require different steering schedules. Understanding why (e.g., refusal is mediated by a single strong direction vs. language requiring distributed features) would inform better steering strategies.

4. **Small vs. large model differences**: Most work uses 7-9B models. Whether decay characteristics change with model scale is unexplored.

5. **Relationship between one-shot and continuous steering**: Belitsky et al. show KV cache steering works as one-shot because of persistence in the attention mechanism. Understanding why activation steering doesn't persist similarly would illuminate the decay mechanism.

---

## Recommendations for Experiment Design

Based on the literature review:

- **Recommended models**: Gemma-2-2B (lightweight, used by Xie), Llama-3.1-8B-Instruct (used by Klerings), or Mistral-7B-Instruct (used by Scalena)
- **Recommended steering method**: CAA (Contrastive Activation Addition) - well-established, with available code
- **Recommended datasets**: CAA behavioral datasets (sycophancy, refusal) for clean A/B evaluation; TruthfulQA for truthfulness; Alpaca for language steering
- **Recommended baselines**: Constant steering, no steering, diminishing (linear), and token-wise decay (Xie formula)
- **Recommended metrics**: Steering accuracy per token position, perplexity, cosine similarity of activations to steering direction over time
- **Key experimental design**: Apply steering vector at steps 0..N only, then measure behavioral conditioning at subsequent steps. Compare with teacher-forcing (feeding the steered model's outputs back without further steering) vs. continued steering. Measure the projection of hidden states onto the steering direction at each token position.
- **Methodological considerations**: Use TransformerLens or hooks for activation extraction. Normalize activations after steering (as recommended by Xie). Control for prompt length and content.
