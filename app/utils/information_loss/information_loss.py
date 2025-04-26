import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class InformationLoss(ABC):
    """
    Parent class for evaluating information loss when simplifying medical text.
    
    This abstract base class defines the interface and common functionality
    for different types of information loss metrics.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the information loss evaluator.
        
        Args:
            name: Short name for the loss type
            description: Text description of this type of information loss
        """
        self.name = name
        self.description = description
        self._loss_value = 0.0  # Internal loss value (0-1)
        self._confidence = 1.0  # Confidence in the loss calculation (0-1)
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except:
            # Fallback to smaller model if the medium one isn't available
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Warning: Using smaller language model. Vector similarity may be less accurate.")
            except:
                print("Error: spaCy model not available.")
                self.nlp = None
    
    @property
    def loss_value(self) -> float:
        """
        Get the calculated loss value (read-only).
        
        Returns:
            Loss value between 0 (no loss) and 1 (complete loss)
        """
        return self._loss_value
    
    @property
    def confidence(self) -> float:
        """
        Get the confidence in the loss calculation (read-only).
        
        Returns:
            Confidence value between 0 (low confidence) and 1 (high confidence)
        """
        return self._confidence
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get evaluation metrics for information loss.
        
        Returns:
            Dictionary with precision, recall, and F1 metrics
        """
        # Interpret loss value as a measure of what was lost
        # Precision: What percentage of the simplified content accurately represents the original
        precision = 1.0 - self._loss_value
        
        # Recall: What percentage of the original content is preserved
        recall = 1.0 - self._loss_value
        
        # F1: Harmonic mean of precision and recall
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss_value": self._loss_value,
            "confidence": self._confidence
        }
    
    @abstractmethod
    def evaluate(self, original_text: str, simplified_text: str) -> float:
        """
        Evaluate information loss between original and simplified text.
        
        Args:
            original_text: Original medical text
            simplified_text: Simplified version of the text
            
        Returns:
            Loss value between 0 (no loss) and 1 (complete loss)
        """
        pass
    