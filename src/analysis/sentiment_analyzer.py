"""
File: sentiment_analyzer.py
Description: Bitcoin headline sentiment analyzer using finBERT
Created by: Renesh Ravi
"""

import logging
import torch
import pandas as pd
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    text: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    probabilities: Dict[str, float]

class SentimentAnalyzer:

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str =
    "auto"):
        self.model_name = model_name
        self.device = self._determine_device(device)

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        self.label_mapping = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE'
        }

        self.bitcoin_label_mapping = {
            'NEGATIVE': 'BEARISH',
            'NEUTRAL': 'NEUTRAL',
            'POSITIVE': 'BULLISH'
        }

        logger.info(f"SentimentAnalyzer initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")


    def _determine_device(self, device: str) -> str :
        """
        Determines whether to use the CPU or GPU component for computation
        :param device:
        :return: The component that will be used for computation.
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("GPU will be used for fast processing.")
            else:
                device = "cpu"
                logger.info("CPU will be used for computing.")
        return device

    def load_model(self):
        """
        Loads the finBERT model.
        """
        if self._is_loaded:
            logger.info("Model is already loaded.")
            return

        logger.info(f"Loading finBERT model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
            self.model = AutoModelForSequenceClassification(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"finBERT model loaded successfully on {self.device}")
            self._is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load funBERT model: {e}")

    def _preprocess_text(self, text: str) -> str:
        """
        Prepares the text for finBERT by stripping whitespace and cleaning
        it up.
        :param text: The text to be preprocessed to be then analyzed by
        finBERT.
        :return: Text that has been processed and ready to be passed to
        finBERT.
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())

        if len(text) > 512:
            text = text[:512]
            logger.debug("The text has been truncated to 512 characters due "
                         "to the maximum length of finBERT.")

        return text

    def analyze_single_headline(self, text: str) -> SentimentResult:
        """
        Analyzes the sentiment for a single headline.
        :param text: The headline to be used for sentiment analysis.
        :return: SentimentResult object containing the sentiment score for
        the headline.
        """
        if not self._is_loaded:
            self.load_model()

        clean_text = self._preprocess_text(text)

        if not clean_text:
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                sentiment_label="NEUTRAL",
                confidence=0.0,
                probabilities={"BEARISH": 0.33, "NEUTRAL": 0.34, "BULLISH":
                    0.33}
            )

        try:
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {key: value.to(self.device) for key, value in
                      inputs.item()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = softmax(logits, dim=-1)

            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities.max().item()
            raw_label = self.model.config.id2label[predicted_class_id]
            sentiment_label = self.label_mapping.get(raw_label, raw_label)
            bitcoin_label = self.bitcoin_label_mapping.get(sentiment_label,
                                                           sentiment_label)

            sentiment_score = self._calculate_sentiment_score(probabilities)

            probs_dict={}
            for i, prob in enumerate(probabilities[0]):
                raw_label = self.model.config.id2label[i]
                sentiment_type = self.label_mapping.get(raw_label, raw_label)
                bitcoin_type = self.bitcoin_label_mapping.get(
                    sentiment_type, sentiment_type)
                probs_dict[bitcoin_type] = prob.item()

            result = SentimentResult(
                text=text,
                sentiment_score=sentiment_score,
                sentiment_label=bitcoin_label,
                confidence=confidence,
                probabilities=probs_dict
            )

            logger.debug(f"Analyzed '{text[:50]}...' -> {bitcoin_label} ("
                         f"{sentiment_score:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error anallyzing the text '{text[:50]}...': {e}")
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                sentiment_label="NEUTRAL",
                confidence=0.0,
                probabilities={"BEARISH": 0.33, "NEUTRAL": 0.34, "BULLISH":
                    0.33}
            )

    def _calculate_sentiment_score(self, probabilities: torch.Tensor) -> (
            float):
        """
        Converts the probability distribution into a single sentiment score
        ranging from -1.0 to 1.0.
        :param probabilities: Probabilities from finBERT analysis.
        :return: Sentiment score ranging from -1.0 to 1.0.
        """
        negative_prob = probabilities[0][0].item()
        neutral_prob = probabilities[0][1].item()
        positive_prob = probabilities[0][2].item()

        return (positive_prob * 1.0) + (neutral_prob * 0.0) + (
            negative_prob * -1.0)
