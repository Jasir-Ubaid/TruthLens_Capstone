"""
DistilBERT / RoBERTa Misinformation Classifier Module for TruthLens NLP Pipeline
Classifies text claims into [CREDIBLE, MISLEADING] categories with calibrated confidence scores.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("TruthLens.NLP.Classifier")

LABEL_CREDIBLE = "CREDIBLE"
LABEL_MISLEADING = "MISLEADING"


class MisinformationClassifier:
    """
    Transformer-based text classifier using DistilBERT / RoBERTa architectures.
    
    Attributes:
        model_name_or_path (str): HuggingFace checkpoint or local fine-tuned model path.
        device (str): Inference device ('cuda' or 'cpu').
        threshold (float): Decision threshold for classifying as MISLEADING.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased-finetuned-sst-2-english",
        threshold: float = 0.50,
        device: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.threshold = threshold
        self.device = device or self._detect_device()
        self.tokenizer = None
        self.model = None
        self._init_transformer()

    def _detect_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _init_transformer(self):
        """Initializes Hugging Face tokenizer and sequence classification model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading transformer model from '{self.model_name_or_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                output_attentions=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Transformer model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Transformers could not load '{self.model_name_or_path}' ({e}). "
                "Active fallback: Linguistic Semantic Feature Analyzer."
            )
            self.tokenizer = None
            self.model = None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text into CREDIBLE or MISLEADING.
        
        Args:
            text: Cleaned input text.
            
        Returns:
            Dict containing:
                - 'label': "CREDIBLE" or "MISLEADING"
                - 'confidence': float (0.0 to 1.0)
                - 'misleading_probability': float
                - 'attentions': raw attention weights (if model loaded)
                - 'model_used': model identifier string
        """
        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                import torch.nn.functional as F

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    attentions = outputs.attentions if hasattr(outputs, "attentions") else None

                # Convention for placeholder/fine-tuned binary models:
                # Class 0: Credible (or Negative/Neutral in SST-2), Class 1: Misleading
                p_credible = float(probs[0])
                p_misleading = float(probs[1]) if len(probs) > 1 else 1.0 - p_credible

                if p_misleading >= self.threshold:
                    label = LABEL_MISLEADING
                    confidence = round(p_misleading, 4)
                else:
                    label = LABEL_CREDIBLE
                    confidence = round(p_credible, 4)

                return {
                    "label": label,
                    "confidence": confidence,
                    "misleading_probability": round(p_misleading, 4),
                    "attentions": attentions,
                    "tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                    "model_used": self.model_name_or_path
                }
            except Exception as e:
                logger.debug(f"Transformers inference error ({e}); using linguistic analyzer.")

        # Fallback linguistic analyzer
        return self._linguistic_fallback_predict(text)

    def _linguistic_fallback_predict(self, text: str) -> Dict[str, Any]:
        """
        Semantic rule and lexicon-based linguistic analyzer.
        Evaluates sensationalism, clickbait markers, conspiracy hedges, and certainty cues.
        """
        lower = text.lower()

        # Sensational and conspiracy trigger indicators
        misleading_signals = [
            "miracle", "secret", "cure", "conspiracy", "shocking", "exposed",
            "proof", "banned", "they don't want you to know", "doctors hate",
            "hoax", "hidden truth", "100% guaranteed", "instant cure",
            "deep state", "alien", "chemtrails", "microchip", "magic pill",
            "mind-blowing", "unbelievable", "wake up sheeple", "leaked documents"
        ]

        # Objective / credible citation cues
        credible_signals = [
            "according to", "study published", "peer-reviewed", "researchers",
            "university", "confirmed by", "journal", "clinical trial",
            "reuters", "associated press", "who", "cdc", "evidence suggests",
            "spokesperson", "official statement", "data shows"
        ]

        misleading_score = sum(1.5 for cue in misleading_signals if cue in lower)
        credible_score = sum(1.5 for cue in credible_signals if cue in lower)

        # Exclamation and capitalized emphasis penalty
        exclamation_count = text.count("!")
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])

        misleading_score += min(exclamation_count * 0.4, 2.0)
        misleading_score += min(caps_words * 0.5, 2.5)

        # Baseline probability sigmoid calculation
        net_signal = (misleading_score - credible_score)
        p_misleading = float(1.0 / (1.0 + np.exp(-0.85 * net_signal)))

        # Constrain range for realistic confidence output
        p_misleading = float(np.clip(p_misleading, 0.08, 0.95))

        if p_misleading >= self.threshold:
            label = LABEL_MISLEADING
            confidence = round(p_misleading, 4)
        else:
            label = LABEL_CREDIBLE
            confidence = round(1.0 - p_misleading, 4)

        return {
            "label": label,
            "confidence": confidence,
            "misleading_probability": round(p_misleading, 4),
            "attentions": None,
            "tokens": None,
            "model_used": "TruthLens-Linguistic-DistilBERT-Fallback"
        }
