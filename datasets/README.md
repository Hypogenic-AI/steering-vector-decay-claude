# Datasets for Steering Vector Decay Research

Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: TruthfulQA (Multiple Choice)

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa` (multiple_choice config)
- **Size**: 817 examples
- **Format**: HuggingFace Dataset
- **Task**: Truthfulness evaluation via multiple-choice questions
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
dataset.save_to_disk("datasets/truthful_qa")
```

### Relevance
Standard benchmark for evaluating truthfulness steering (Li et al. 2023 ITI).
Used to measure whether a steering vector applied at initial tokens maintains
truthfulness conditioning throughout multi-token generation.

---

## Dataset 2: Alpaca (General QA)

### Overview
- **Source**: HuggingFace `tatsu-lab/alpaca`
- **Size**: 1000 examples (subset of 52K)
- **Format**: HuggingFace Dataset
- **Task**: Instruction following / QA
- **License**: CC BY-NC 4.0

### Download Instructions
```python
from datasets import load_dataset
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
alpaca_subset = alpaca.select(range(1000))
alpaca_subset.save_to_disk("datasets/alpaca_subset")
```

### Relevance
Used by Scalena et al. (2024) for language steering experiments. Good general-purpose
dataset for measuring steering decay across varied prompt types.

---

## Dataset 3: BeaverTails (Safety)

### Overview
- **Source**: HuggingFace `PKU-Alignment/BeaverTails`
- **Size**: 500 examples (subset of 30k_test split)
- **Format**: HuggingFace Dataset
- **Task**: Safety evaluation - prompts designed to elicit unsafe responses
- **License**: CC BY-NC 4.0

### Download Instructions
```python
from datasets import load_dataset
bt = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
bt_subset = bt.select(range(500))
bt_subset.save_to_disk("datasets/beavertails_subset")
```

### Relevance
Used by Scalena et al. (2024) for safety steering experiments. Key dataset for
studying whether safety steering applied at initial tokens decays over generation.

---

## Dataset 4: CAA Behavioral Datasets (from cloned repo)

### Overview
- **Source**: `code/CAA/datasets/` (cloned from github.com/nrimsky/CAA)
- **Size**: Multiple JSON files covering 7 behavioral categories
- **Format**: JSON files with A/B multiple choice format
- **Categories**: sycophancy, hallucination, refusal, survival-instinct,
  corrigible-neutral-HHH, myopic-reward, coordinate-other-ais
- **License**: MIT

### Loading
```python
import json
with open("code/CAA/datasets/test/sycophancy/sycophancy_test.json") as f:
    data = json.load(f)
```

### Relevance
Original behavioral datasets from Rimsky et al. (2024) CAA paper. Multiple-choice
format allows precise measurement of steering vector influence. Can be used to
test whether steering applied only at the first few tokens maintains its effect
on the final answer choice.

---

## Notes
- All HuggingFace datasets can be re-downloaded with the scripts above
- CAA datasets are available in the cloned repo at `code/CAA/datasets/`
- Sample files (*_samples.json) are committed for reference
