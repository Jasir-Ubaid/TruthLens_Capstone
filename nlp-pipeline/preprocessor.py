"""
Text Preprocessor Module for TruthLens NLP Pipeline
Cleans and normalizes incoming claims, social posts, and news snippets,
and segments text into tokens while preserving span offsets for explainability highlighting.
"""

import re
import html
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("TruthLens.NLP.Preprocessor")

# Common web/social noise patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")
MENTION_HASHTAG_PATTERN = re.compile(r"[@#]\w+")


class TextPreprocessor:
    """
    Cleans raw text and generates token span maps for explainability highlighting.
    """

    def __init__(self, lowercase: bool = False, remove_urls: bool = False):
        self.lowercase = lowercase
        self.remove_urls = remove_urls

    def clean(self, raw_text: str) -> str:
        """
        Cleans and normalizes input text.
        
        Args:
            raw_text: Raw input claim or article snippet.
            
        Returns:
            Normalized clean text string.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""

        # Decode HTML entities (e.g. &amp; -> &)
        text = html.unescape(raw_text)

        # Remove HTML tags if present
        text = HTML_TAG_PATTERN.sub(" ", text)

        # Handle URLs
        if self.remove_urls:
            text = URL_PATTERN.sub("", text)

        # Collapse excess whitespace while preserving standard punctuation
        text = WHITESPACE_PATTERN.sub(" ", text).strip()

        if self.lowercase:
            text = text.lower()

        return text

    def tokenize_with_spans(self, text: str) -> List[Dict[str, Any]]:
        """
        Tokenizes text while tracking exact start and end character offsets.
        Offsets are essential for reconstructing highlighted HTML phrases.
        
        Returns:
            List of dicts: [{'token': str, 'start': int, 'end': int}]
        """
        tokens = []
        # Match word tokens, preserving contraction components
        for match in re.finditer(r"\b[\w'-]+\b|[^\w\s]", text):
            token_str = match.group(0)
            tokens.append({
                "token": token_str,
                "start": match.start(),
                "end": match.end()
            })
        return tokens

    def split_sentences(self, text: str) -> List[str]:
        """Splits multi-sentence text into clean sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
