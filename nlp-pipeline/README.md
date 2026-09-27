# TruthLens — NLP Misinformation Detection Pipeline (NLP Team)

This module handles text normalization, tokenization, sequence classification using fine-tuned Transformers (DistilBERT / RoBERTa), and token-level attention explainability highlighting.

---

## 📁 Architecture & File Layout

| File | Purpose |
| :--- | :--- |
| `preprocessor.py` | Normalizes whitespace, strips noise/HTML, preserves character offsets for exact phrase highlighting. |
| `classifier.py` | Transformer classification pipeline. Computes probability distribution over `[CREDIBLE, MISLEADING]` labels. Supports Hugging Face checkpoints or local fine-tuned models. |
| `explainer.py` | Extracts self-attention weights from Transformer layers (or semantic gradient fallback), computes token importance scores, and renders styled HTML with highlights. |
| `pipeline.py` | High-level orchestrator class `NLPPipeline` tying together preprocessing, classification, and explainability. |

---

## 🚀 Quickstart for NLP Team Member

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Standalone Testing in Python
```python
from pipeline import NLPPipeline

# Initialize with placeholder or fine-tuned model
pipeline = NLPPipeline(
    model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
)

result = pipeline.analyze("Miracle herbal drink cures diabetes in 48 hours without insulin!")

print("Verdict:", result["verdict"])
print("Confidence:", result["confidence"])
print("Top Phrases:", result["highlighted_phrases"])
print("Rendered HTML:", result["highlighted_html"])
```

---

## 🛠️ How to Plug In Fine-Tuned Weights

When you fine-tune DistilBERT or RoBERTa on datasets like **LIAR**, **FakeNewsNet**, or **WELFake**:

1. Save your Hugging Face model and tokenizer to a directory:
   ```python
   model.save_pretrained("./checkpoints/truthlens-distilbert")
   tokenizer.save_pretrained("./checkpoints/truthlens-distilbert")
   ```
2. Pass the folder path or HuggingFace Model Hub ID to `NLPPipeline`:
   ```python
   pipeline = NLPPipeline(
       model_name_or_path="./checkpoints/truthlens-distilbert"
   )
   ```
3. The pipeline will automatically inspect the model's output attentions, calculate token importances, and produce the explainability output.
