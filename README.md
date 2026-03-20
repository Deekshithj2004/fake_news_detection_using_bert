# Fake News Detection — Assignment 2

**Author:** Deekshith  
**Tool:** Python (Google Colab, T4 GPU)  
**Version:** HuggingFace Trainer API (Version 3)

---

## Problem Statement

Build a binary text classification system to automatically detect whether a news headline or article is **Real (0)** or **Fake (1)**. The model should learn linguistic patterns that distinguish credible journalism from misinformation, and be deployable as an interactive web app for real-time inference.

---

## Approach

The pipeline is structured across seven steps:

1. **Data Loading & Cleaning** — A custom dataset of 20 labelled news headlines (10 real, 10 fake) was constructed. Text is cleaned by stripping URLs and collapsing whitespace. The dataset is designed to be swappable with any `train.csv` file.

2. **HuggingFace DatasetDict** — Data is split into train/validation/test sets (stratified) and wrapped in a `DatasetDict` — the native HuggingFace format for use with the Trainer API.

3. **Tokenization with Dynamic Padding** — The `distilbert-base-uncased` tokenizer truncates inputs to 256 tokens. `DataCollatorWithPadding` pads each batch only to its own maximum length, which is faster than fixed-length padding.

4. **Metrics Callback** — A `compute_metrics` function computes **Accuracy** and **weighted F1** at the end of each evaluation epoch using the `evaluate` library.

5. **Training via HuggingFace Trainer** — The `Trainer` API handles the full training loop (no manual `for epoch` loop). Key settings include cosine LR scheduling, warmup, weight decay, mixed precision (fp16 on GPU), and `EarlyStoppingCallback` with patience of 2 epochs to prevent overfitting.

6. **Evaluation** — After training, the best checkpoint is evaluated on the held-out test set. A full `classification_report` and both raw and normalised confusion matrices are produced.

7. **Gradio Deployment** — The saved model is loaded and served via a Gradio web interface where users can type any headline and receive a real-time prediction with confidence scores.

---

## Model Used

**DistilBERT** (`distilbert-base-uncased`)  
`AutoModelForSequenceClassification` with `num_labels=2`

DistilBERT is a distilled (compressed) version of BERT — 40% smaller and 60% faster while retaining ~97% of BERT's language understanding capability. It is well-suited for text classification tasks on limited compute budgets such as a free Colab T4 GPU.

| Hyperparameter | Value |
|---|---|
| Max sequence length | 256 tokens |
| Batch size (train) | 16 |
| Epochs | 3 (with early stopping) |
| Learning rate | 2e-5 |
| LR scheduler | Cosine with warmup (10%) |
| Weight decay | 0.01 |
| Precision | fp16 (GPU) |

---

## Metrics

Evaluation is performed on the held-out test set using the best checkpoint (selected by validation accuracy):

| Metric | Description |
|---|---|
| **Accuracy** | Overall fraction of correctly classified samples |
| **Weighted F1** | F1 averaged by class support — handles class imbalance |
| **Precision / Recall** | Per-class breakdown via `classification_report` |
| **Confusion Matrix** | Raw counts and normalised percentages |

Training curves (train loss, validation loss, accuracy, F1 per step) are plotted from `trainer.state.log_history`.

> Note: With only 20 samples, reported metrics are illustrative. Real-world performance should be evaluated on a larger benchmark dataset such as LIAR or FakeNewsNet.

---

## Improvements

- **Scale up the dataset** — 20 samples is far too small for fine-tuning a transformer reliably. Loading a real dataset (e.g. via `datasets.load_dataset('liar')`) would be the single highest-impact improvement.
- **Longer training with larger dataset** — With more data, increasing epochs and reducing the learning rate would allow the model to converge more robustly.
- **Experiment with larger models** — `bert-base-uncased` or `roberta-base` may outperform DistilBERT on nuanced fake news patterns.
- **Class imbalance handling** — Adding class weights to the loss function or oversampling minority classes would help if the real dataset is imbalanced.
- **Evaluation on unseen sources** — Testing on data from different time periods or news domains tests true generalisation, not just in-distribution accuracy.
- **Explainability** — Integrating attention visualisation or SHAP values would make predictions more interpretable and trustworthy.
- **Gradio sharing** — The Gradio app uses `share=True` (temporary Colab link). Deploying to HuggingFace Spaces would make it permanently accessible.

---

## Key Learnings

- **HuggingFace Trainer simplifies training significantly** — Replacing a manual training loop with `Trainer.train()` reduces boilerplate and automatically handles evaluation, checkpointing, and early stopping.
- **Dynamic padding is more efficient than fixed padding** — `DataCollatorWithPadding` reduces wasted computation on short sequences within a batch.
- **Early stopping prevents wasted compute** — With `patience=2`, training halts automatically if validation accuracy does not improve, saving GPU time.
- **`load_best_model_at_end=True` is critical** — Without it, the final checkpoint (not the best one) would be used for evaluation, potentially giving worse results.
- **Small datasets and transformer fine-tuning are a poor match** — Pre-trained transformers need sufficient labelled examples to update their weights meaningfully; 20 samples will almost certainly lead to overfitting or unreliable metrics.
- **Gradio makes deployment trivial** — Wrapping the inference function in `gr.Interface` provides a shareable demo with zero frontend code.

---

## Project Structure

```
assignment2_colab.ipynb    # Main notebook with all 7 steps
README.md                  # This file
```

## Dependencies

```
transformers
datasets
accelerate
evaluate
gradio
scikit-learn
seaborn
matplotlib
pandas
torch
```

## Quick Start

```
Runtime → Change runtime type → T4 GPU
Run all cells in order
```
