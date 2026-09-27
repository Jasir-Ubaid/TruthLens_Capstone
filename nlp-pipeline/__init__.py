"""
TruthLens NLP Misinformation Detection Pipeline
Modular package for text preprocessing, Transformer-based classification (DistilBERT/RoBERTa),
and attention-based phrase explainability highlighting.
"""

from .pipeline import NLPPipeline

__all__ = ["NLPPipeline"]
