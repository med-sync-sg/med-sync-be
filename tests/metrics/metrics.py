# tests/metrics/nlp_metrics.py
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter

def calculate_entity_recognition_metrics(correct_entities: List[Dict], 
                                        predicted_entities: List[Dict]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 score and error rate for entity recognition
    
    Args:
        correct_entities: List of standard entities with 'text' and 'label' fields
        predicted_entities: List of predicted entities with 'text' and 'label' fields
        
    Returns:
        Dictionary with precision, recall, F1 score and error rate
    """
    # Convert to lowercase for comparison
    correct_texts = [entity['text'].lower() for entity in correct_entities]
    pred_texts = [entity['text'].lower() for entity in predicted_entities]
    
    # Count true positives, false positives, false negatives
    true_positives = sum(1 for text in pred_texts if text in correct_texts)
    false_positives = sum(1 for text in pred_texts if text not in correct_texts)
    false_negatives = sum(1 for text in correct_texts if text not in pred_texts)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Error rate (1 - F1)
    error_rate = 1.0 - f1
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "error_rate": error_rate,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def calculate_classification_accuracy(correct_entities: List[Dict], 
                                     predicted_entities: List[Dict]) -> Dict[str, float]:
    """
    Calculate accuracy and error rate for entity classification
    
    Args:
        correct_entities: List of correct standard entities with 'text' and 'label' fields
        predicted_entities: List of predicted entities with 'text' and 'label' fields
        
    Returns:
        Dictionary with accuracy, error rate and confusion matrix data
    """
    # Create a dictionary mapping entity text to label for easy lookup
    correct_dict = {entity['text'].lower(): entity['label'] for entity in correct_entities}
    
    correct = 0
    incorrect = 0
    confusion = Counter()
    
    # Compare labels for matching entities
    for pred in predicted_entities:
        pred_text = pred['text'].lower()
        if pred_text in correct_dict:
            correct_label = correct_dict[pred_text]
            pred_label = pred['label']
            
            if correct_label == pred_label:
                correct += 1
            else:
                incorrect += 1
                confusion[(correct_label, pred_label)] += 1
    
    # Calculate metrics
    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0
    error_rate = 1.0 - accuracy
    
    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "correct": correct,
        "incorrect": incorrect,
        "confusion_counts": dict(confusion)
    }

def evaluate_keyword_extraction(correct_keywords: List[Dict],
                               predicted_keywords: List[Dict]) -> Dict[str, float]:
    """
    Evaluate keyword extraction quality
    
    Args:
        correct_keywords: List of expected keyword dictionaries with 'term', 'modifiers', 'quantities'
        predicted_keywords: List of extracted keyword dictionaries
        
    Returns:
        Dictionary with metrics for term extraction, modifier extraction, etc.
    """
    # Extract terms for comparison
    correct_terms = set(kw['term'].lower() for kw in correct_keywords)
    pred_terms = set(kw['term'].lower() for kw in predicted_keywords)
    
    # Term extraction metrics
    term_true_positives = len(correct_terms.intersection(pred_terms))
    term_false_positives = len(pred_terms - correct_terms)
    term_false_negatives = len(correct_terms - pred_terms)
    
    term_precision = term_true_positives / len(pred_terms) if pred_terms else 0
    term_recall = term_true_positives / len(correct_terms) if correct_terms else 0
    term_f1 = 2 * term_precision * term_recall / (term_precision + term_recall) if (term_precision + term_recall) > 0 else 0
    term_error_rate = 1.0 - term_f1
    
    # Create dictionaries for modifier and quantity comparison
    correct_dict = {kw['term'].lower(): {'modifiers': set(m.lower() for m in kw.get('modifiers', [])),
                                     'quantities': set(q.lower() for q in kw.get('quantities', []))}
                for kw in correct_keywords}
    
    # Metrics for modifiers and quantities
    modifier_precision = 0
    modifier_recall = 0
    quantity_precision = 0
    quantity_recall = 0
    
    # Count matches
    matched_terms = 0
    
    for pred_kw in predicted_keywords:
        pred_term = pred_kw['term'].lower()
        if pred_term in correct_dict:
            matched_terms += 1
            
            # Compare modifiers
            correct_modifiers = correct_dict[pred_term]['modifiers']
            pred_modifiers = set(m.lower() for m in pred_kw.get('modifiers', []))
            
            if correct_modifiers and pred_modifiers:
                mod_precision = len(correct_modifiers.intersection(pred_modifiers)) / len(pred_modifiers)
                mod_recall = len(correct_modifiers.intersection(pred_modifiers)) / len(correct_modifiers)
                modifier_precision += mod_precision
                modifier_recall += mod_recall
            
            # Compare quantities
            correct_quantities = correct_dict[pred_term]['quantities']
            pred_quantities = set(q.lower() for q in pred_kw.get('quantities', []))
            
            if correct_quantities and pred_quantities:
                quant_precision = len(correct_quantities.intersection(pred_quantities)) / len(pred_quantities)
                quant_recall = len(correct_quantities.intersection(pred_quantities)) / len(correct_quantities)
                quantity_precision += quant_precision
                quantity_recall += quant_recall
    
    # Average precision and recall for modifiers and quantities
    if matched_terms > 0:
        modifier_precision /= matched_terms
        modifier_recall /= matched_terms
        quantity_precision /= matched_terms
        quantity_recall /= matched_terms
    
    # Calculate F1 scores
    modifier_f1 = 2 * modifier_precision * modifier_recall / (modifier_precision + modifier_recall) if (modifier_precision + modifier_recall) > 0 else 0
    quantity_f1 = 2 * quantity_precision * quantity_recall / (quantity_precision + quantity_recall) if (quantity_precision + quantity_recall) > 0 else 0
    
    return {
        "term_metrics": {
            "precision": term_precision,
            "recall": term_recall,
            "f1_score": term_f1,
            "error_rate": term_error_rate
        },
        "modifier_metrics": {
            "precision": modifier_precision,
            "recall": modifier_recall,
            "f1_score": modifier_f1,
            "error_rate": 1.0 - modifier_f1
        },
        "quantity_metrics": {
            "precision": quantity_precision,
            "recall": quantity_recall,
            "f1_score": quantity_f1,
            "error_rate": 1.0 - quantity_f1
        }
    }