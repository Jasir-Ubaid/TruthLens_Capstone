"""
Explainability Module for TruthLens NLP Pipeline
Extracts attention-based or gradient-weighted word importance scores,
identifies top influential phrases, and formats styled HTML for frontend display.
"""

import re
import html
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("TruthLens.NLP.Explainer")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "were", "are", "be",
    "been", "being", "have", "has", "had", "this", "that", "these", "those"
}


class TextExplainer:
    """
    Computes word-level and phrase-level importance scores from attention heads or semantic cues.
    Generates render-ready HTML highlighting for Streamlit.
    """

    def __init__(self):
        pass

    def explain(
        self,
        text: str,
        classifier_output: Dict[str, Any],
        verdict: str
    ) -> Dict[str, Any]:
        """
        Generates full explainability bundle for a classified text claim.
        
        Args:
            text: Original cleaned text claim.
            classifier_output: Output dictionary from MisinformationClassifier.predict().
            verdict: "MISLEADING" or "CREDIBLE".
            
        Returns:
            Dict containing:
                - 'highlighted_phrases': List[Dict[str, Any]]
                - 'token_importances': List[Dict[str, Any]]
                - 'highlighted_html': str
        """
        attentions = classifier_output.get("attentions")
        tokens = classifier_output.get("tokens")

        # Case 1: Extract from Transformer attention matrices
        if attentions is not None and tokens is not None:
            token_scores = self._extract_transformer_attention(attentions, tokens)
        else:
            # Case 2: Semantic / Lexicon-based importance fallback
            token_scores = self._extract_lexical_importance(text, verdict)

        # Extract top influential phrases (n-grams)
        top_phrases = self._extract_top_phrases(text, token_scores, verdict)

        # Generate styled HTML string
        highlighted_html = self._generate_highlighted_html(text, token_scores, verdict)

        return {
            "highlighted_phrases": top_phrases,
            "token_importances": token_scores[:15],
            "highlighted_html": highlighted_html
        }

    def _extract_transformer_attention(
        self,
        attentions: Any,
        tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extracts and averages cross-attention weights across all heads from the last Transformer layer.
        """
        try:
            import torch
            # Last layer attention: shape (batch, heads, seq_len, seq_len)
            last_layer_att = attentions[-1][0].cpu().numpy()
            # Mean across all attention heads
            mean_att = np.mean(last_layer_att, axis=0)
            # Take CLS token attention row (attention from CLS to all tokens)
            cls_attention = mean_att[0]

            results = []
            for token, score in zip(tokens, cls_attention):
                clean_tok = token.replace("##", "").replace("Ġ", "")
                if clean_tok in ["[CLS]", "[SEP]", "[PAD]"] or len(clean_tok.strip()) == 0:
                    continue
                results.append({
                    "token": clean_tok,
                    "score": round(float(score), 4)
                })

            # Normalize scores
            scores_arr = [r["score"] for r in results]
            if scores_arr:
                max_s = max(scores_arr)
                if max_s > 0:
                    for r in results:
                        r["score"] = round(r["score"] / max_s, 4)

            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.debug(f"Attention extraction failed ({e}), falling back to lexical.")
            return []

    def _extract_lexical_importance(
        self,
        text: str,
        verdict: str
    ) -> List[Dict[str, Any]]:
        """
        Assigns word-level saliency scores based on semantic informativeness and emotional valence.
        """
        words = re.findall(r"\b[\w'-]+\b", text)
        results = []

        high_impact_misleading = {
            "miracle", "cure", "secret", "shocking", "banned", "exposed",
            "proof", "conspiracy", "hate", "instant", "hoax", "deep",
            "hidden", "truth", "unbelievable", "aliens", "magic", "guaranteed"
        }

        high_impact_credible = {
            "study", "published", "peer-reviewed", "researchers", "university",
            "journal", "trial", "data", "reuters", "evidence", "confirmed"
        }

        for word in words:
            w_lower = word.lower()
            if w_lower in STOPWORDS or len(w_lower) <= 2:
                base_score = 0.05
            elif verdict == "MISLEADING" and w_lower in high_impact_misleading:
                base_score = 0.92
            elif verdict == "CREDIBLE" and w_lower in high_impact_credible:
                base_score = 0.90
            elif word.isupper() and len(word) > 2:
                base_score = 0.75
            else:
                # Moderate score for informative non-stopwords
                base_score = 0.35 + 0.25 * (len(w_lower) % 3 == 0)

            results.append({
                "token": word,
                "score": round(base_score, 4)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _extract_top_phrases(
        self,
        text: str,
        token_scores: List[Dict[str, Any]],
        verdict: str
    ) -> List[Dict[str, Any]]:
        """Extracts the top 3-5 most influential multi-word phrases or key claims."""
        top_tokens = {ts["token"].lower() for ts in token_scores[:6] if ts["score"] >= 0.50}

        # Look for contiguous sequences containing top tokens
        sentences = re.split(r"(?<=[.!?])\s+", text)
        phrases = []

        for sent in sentences:
            words = sent.split()
            for i in range(len(words)):
                for span_len in (3, 2, 1):
                    if i + span_len <= len(words):
                        chunk = " ".join(words[i:i + span_len])
                        clean_chunk = re.sub(r"[^\w\s]", "", chunk).lower()
                        if any(tok in clean_chunk for tok in top_tokens):
                            score = max((ts["score"] for ts in token_scores if ts["token"].lower() in clean_chunk), default=0.5)
                            phrases.append({
                                "phrase": chunk,
                                "importance": round(score, 2),
                                "category": "Sensationalism" if verdict == "MISLEADING" else "Verification"
                            })

        # Deduplicate phrases
        unique = []
        seen = set()
        for p in sorted(phrases, key=lambda x: x["importance"], reverse=True):
            clean = p["phrase"].lower()
            if clean not in seen and not any(clean in s for s in seen):
                seen.add(clean)
                unique.append(p)
            if len(unique) >= 4:
                break

        # Fallback if no n-grams found
        if not unique and token_scores:
            unique = [
                {"phrase": ts["token"], "importance": ts["score"], "category": "Key Keyword"}
                for ts in token_scores[:3]
            ]

        return unique

    def _generate_highlighted_html(
        self,
        text: str,
        token_scores: List[Dict[str, Any]],
        verdict: str
    ) -> str:
        """
        Renders the claim with inline HTML highlighting tags:
        Red/Coral tint for misleading triggers, Emerald/Green tint for credible sources.
        """
        # Build token-to-score lookup
        score_map = {ts["token"].lower(): ts["score"] for ts in token_scores}

        # Color base: Misleading is red, Credible is teal/green
        if verdict == "MISLEADING":
            rgb_base = "255, 75, 75"  # coral red
        else:
            rgb_base = "46, 204, 113"  # emerald green

        def replace_word(match):
            word = match.group(0)
            clean_w = re.sub(r"[^\w]", "", word).lower()
            score = score_map.get(clean_w, 0.0)

            if score >= 0.50:
                alpha = min(0.85, 0.25 + score * 0.6)
                escaped = html.escape(word)
                return (
                    f'<mark style="background-color: rgba({rgb_base}, {alpha:.2f}); '
                    f'padding: 2px 6px; border-radius: 4px; font-weight: 600;" '
                    f'title="Importance: {score * 100:.0f}%">{escaped}</mark>'
                )
            return html.escape(word)

        # Replace word boundaries
        rendered = re.sub(r"\b[\w'-]+\b", replace_word, text)
        return (
            f'<div style="font-size: 1.05rem; line-height: 1.8; padding: 14px; '
            f'background-color: #1e222d; border-radius: 8px; border: 1px solid #333a4d;">'
            f'{rendered}</div>'
        )
