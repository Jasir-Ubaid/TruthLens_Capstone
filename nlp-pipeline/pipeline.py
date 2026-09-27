"""
TruthLens NLP Misinformation Detection Pipeline Orchestrator
Coordinates preprocessing, DistilBERT / RoBERTa classification,
and attention-based phrase explainability.
"""

import logging
from typing import Dict, Any, Optional

from .preprocessor import TextPreprocessor
from .classifier import MisinformationClassifier
from .explainer import TextExplainer

logger = logging.getLogger("TruthLens.NLP.Pipeline")


class NLPPipeline:
    """
    End-to-End Misinformation Detection Pipeline.
    
    Workflow:
        1. Preprocess & normalize raw text claim / headline.
        2. Sequence classification with DistilBERT / RoBERTa model.
        3. Explainability extraction (attention weights / lexical triggers).
        4. Render formatted HTML highlighting for the frontend.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased-finetuned-sst-2-english",
        threshold: float = 0.50
    ):
        self.preprocessor = TextPreprocessor(lowercase=False, remove_urls=False)
        self.classifier = MisinformationClassifier(
            model_name_or_path=model_name_or_path,
            threshold=threshold
        )
        self.explainer = TextExplainer()
        self.model_name = f"TruthLens-{model_name_or_path.split('/')[-1].upper()}"

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Executes end-to-end misinformation analysis on a text input.
        
        Args:
            text: Raw claim, headline, or excerpt.
            
        Returns:
            Structured dictionary with verdict, confidence, and explainability outputs.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        clean_text = self.preprocessor.clean(text)
        logger.info(f"Running NLP classification on: '{clean_text[:60]}...'")

        # Step 1: Classification
        clf_result = self.classifier.predict(clean_text)
        verdict = clf_result["label"]
        confidence = clf_result["confidence"]

        # Step 2: Explainability
        explanation = self.explainer.explain(clean_text, clf_result, verdict)

        summary = (
            f"Evaluated claim ({len(clean_text.split())} words). "
            f"Assigned verdict: {verdict} ({confidence * 100:.1f}% confidence). "
            f"Identified {len(explanation['highlighted_phrases'])} key influential phrases."
        )

        return {
            "claim_text": clean_text,
            "verdict": verdict,
            "confidence": confidence,
            "misleading_probability": clf_result.get("misleading_probability", 0.5),
            "model_name": self.model_name,
            "highlighted_phrases": explanation["highlighted_phrases"],
            "token_importances": explanation["token_importances"],
            "highlighted_html": explanation["highlighted_html"],
            "summary": summary
        }
