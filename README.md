# Cross-Domain Sentiment Classification with RoBERTa

Fine-tunes a RoBERTa-base transformer for binary sentiment classification, combining movie and product review datasets to evaluate cross-domain generalization.

## Overview

This project fine-tunes `roberta-base` (via Hugging Face Transformers) on a **combined IMDB + Amazon reviews dataset**, then evaluates specifically on the Amazon subset — testing how well a model trained on mixed-domain sentiment data generalizes to a target domain.

## How It Works

**Preprocessing** (`preprocess_function`)
- Tokenizes raw text using the RoBERTa tokenizer
- Applies truncation and fixed-length padding (max length 128 tokens)

**Model initialization** (`init_model`)
- Loads `roberta-base` with a sequence classification head configured for 2 output labels (positive/negative sentiment)

**Training** (`train_model`)
- Combines the IMDB and Amazon datasets into a single training set using `ConcatDataset`
- Uses Hugging Face's `Trainer` API with:
  - Learning rate: 1e-5
  - Batch size: 16
  - 3 training epochs
  - Weight decay: 0.01 (regularization)
  - 10% warmup ratio (gradually ramps up learning rate at the start of training)
- **Evaluates exclusively on the Amazon dataset**, even though training includes both — the goal is that the model generalizes well on Amazon product reviews
- Tracks accuracy via a custom `compute_metrics` function (argmax over logits vs. labels)

## Usage

Requires pre-tokenized `imdb_dataset` and `amazon_dataset` objects (each yielding `input_ids`, `attention_mask`, and `labels`) as input to `train_model`.

```python
model = init_model()
model = train_model(model, imdb_dataset, amazon_dataset)
```

## Notes

- Since evaluation is Amazon-only, this setup is really testing **cross-domain transfer** (does adding IMDB data help/hurt Amazon sentiment classification?) rather than in-domain performance — worth stating explicitly if you write this up further, since it's not obvious from the code alone.
- `report_to="none"` disables experiment tracking (e.g., W&B, TensorBoard) — worth enabling for deeper training diagnostics if iterating further.
