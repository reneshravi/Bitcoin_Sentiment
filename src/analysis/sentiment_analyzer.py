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
