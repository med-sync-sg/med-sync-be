"""
Test module for evaluating NLP pipeline performance

This module provides functions to test and evaluate the NLP pipeline
performance against gold standard data.
"""

import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import io
import base64
from collections import Counter
import numpy as np

# Import NLP pipeline components
from app.utils.nlp.spacy_utils import process_text, find_medical_modifiers
from app.db.data_loader import classify_text_category
from app.utils.nlp.summarizer import generate_summary

# Configure logger
logger = logging.getLogger(__name__)

def calculate_entity_metrics(gold_entities: List[Dict[str, Any]],
                             predicted_entities: List[Dict[str, Any]],
                             case_sensitive: bool = False) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 for entity extraction
    
    Args:
        gold_entities: List of gold standard entities with at least 'text' and 'label' keys
        predicted_entities: List of predicted entities with at least 'text' and 'label' keys
        case_sensitive: Whether to consider case in matching
        
    Returns:
        Dictionary with precision, recall, F1 metrics and detailed statistics
    """
    # Process entities to handle case sensitivity
    def normalize_text(text):
        return text if case_sensitive else text.lower()
    
    gold_texts = [normalize_text(e['text']) for e in gold_entities]
    pred_texts = [normalize_text(e['text']) for e in predicted_entities]
    
    # Calculate exact match metrics
    true_positives = sum(1 for text in pred_texts if text in gold_texts)
    false_positives = sum(1 for text in pred_texts if text not in gold_texts)
    false_negatives = sum(1 for text in gold_texts if text not in pred_texts)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Also calculate label-specific metrics if labels are provided
    label_metrics = {}
    if all('label' in e for e in gold_entities) and all('label' in e for e in predicted_entities):
        gold_items = [(normalize_text(e['text']), e['label']) for e in gold_entities]
        pred_items = [(normalize_text(e['text']), e['label']) for e in predicted_entities]
        
        # Group metrics by label
        for label in set(e['label'] for e in gold_entities):
            label_gold = [(text, l) for text, l in gold_items if l == label]
            label_pred = [(text, l) for text, l in pred_items if l == label]
            
            label_tp = sum(1 for item in label_pred if item in label_gold)
            label_fp = sum(1 for item in label_pred if item not in label_gold)
            label_fn = sum(1 for item in label_gold if item not in label_pred)
            
            if label_tp + label_fp > 0:
                label_precision = label_tp / (label_tp + label_fp)
            else:
                label_precision = 0
                
            if label_tp + label_fn > 0:
                label_recall = label_tp / (label_tp + label_fn)
            else:
                label_recall = 0
                
            if label_precision + label_recall > 0:
                label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall)
            else:
                label_f1 = 0
                
            label_metrics[label] = {
                "precision": label_precision,
                "recall": label_recall,
                "f1": label_f1,
                "support": len(label_gold)
            }
    
    metrics = {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_rate": 1.0 - f1
        },
        "counts": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "gold_count": len(gold_texts),
            "pred_count": len(pred_texts)
        },
        "by_label": label_metrics
    }
    
    return metrics

def evaluate_medical_term_extraction(gold_data: List[Dict[str, Any]],
                                     predicted_data: List[Dict[str, Any]],
                                     term_key: str = 'term') -> Dict[str, Any]:
    """
    Evaluate medical term extraction with modifiers
    
    Args:
        gold_data: List of gold standard term dictionaries
        predicted_data: List of predicted term dictionaries
        term_key: Key in dictionaries for the term text
        
    Returns:
        Dictionary with term and modifier metrics
    """
    # Extract terms from both gold and predicted data
    gold_terms = set(d[term_key].lower() for d in gold_data if term_key in d)
    pred_terms = set(d[term_key].lower() for d in predicted_data if term_key in d)
    
    # Calculate basic term metrics
    term_tp = len(gold_terms.intersection(pred_terms))
    term_fp = len(pred_terms - gold_terms)
    term_fn = len(gold_terms - pred_terms)
    
    term_precision = term_tp / (term_tp + term_fp) if (term_tp + term_fp) > 0 else 0
    term_recall = term_tp / (term_tp + term_fn) if (term_tp + term_fn) > 0 else 0
    term_f1 = 2 * term_precision * term_recall / (term_precision + term_recall) if (term_precision + term_recall) > 0 else 0
    
    # Create mapping of terms to modifiers for both datasets
    gold_term_modifiers = {d[term_key].lower(): d.get('modifiers', []) for d in gold_data if term_key in d}
    pred_term_modifiers = {d[term_key].lower(): d.get('modifiers', []) for d in predicted_data if term_key in d}
    
    # Calculate modifier metrics for terms that exist in both datasets
    common_terms = gold_terms.intersection(pred_terms)
    modifier_metrics = {}
    
    for term in common_terms:
        gold_mods = set(m.lower() for m in gold_term_modifiers.get(term, []))
        pred_mods = set(m.lower() for m in pred_term_modifiers.get(term, []))
        
        mod_tp = len(gold_mods.intersection(pred_mods))
        mod_fp = len(pred_mods - gold_mods)
        mod_fn = len(gold_mods - pred_mods)
        
        if mod_tp + mod_fp > 0:
            mod_precision = mod_tp / (mod_tp + mod_fp)
        else:
            mod_precision = 0
            
        if mod_tp + mod_fn > 0:
            mod_recall = mod_tp / (mod_tp + mod_fn)
        else:
            mod_recall = 0
            
        if mod_precision + mod_recall > 0:
            mod_f1 = 2 * mod_precision * mod_recall / (mod_precision + mod_recall)
        else:
            mod_f1 = 0
            
        modifier_metrics[term] = {
            "precision": mod_precision,
            "recall": mod_recall,
            "f1": mod_f1,
            "gold_modifiers": list(gold_mods),
            "pred_modifiers": list(pred_mods)
        }
    
    # Calculate overall modifier metrics
    if modifier_metrics:
        avg_mod_precision = sum(m["precision"] for m in modifier_metrics.values()) / len(modifier_metrics)
        avg_mod_recall = sum(m["recall"] for m in modifier_metrics.values()) / len(modifier_metrics)
        avg_mod_f1 = sum(m["f1"] for m in modifier_metrics.values()) / len(modifier_metrics)
    else:
        avg_mod_precision = 0
        avg_mod_recall = 0
        avg_mod_f1 = 0
    
    # Compile the results
    metrics = {
        "term_extraction": {
            "precision": term_precision,
            "recall": term_recall,
            "f1": term_f1,
            "support": len(gold_terms)
        },
        "modifier_extraction": {
            "overall": {
                "precision": avg_mod_precision,
                "recall": avg_mod_recall,
                "f1": avg_mod_f1,
            },
            "by_term": modifier_metrics
        }
    }
    
    return metrics

def evaluate_text_classification(gold_labels: List[str],
                                 predicted_labels: List[str],
                                 label_set: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate text classification performance
    
    Args:
        gold_labels: List of gold standard labels
        predicted_labels: List of predicted labels
        label_set: Optional list of all possible labels
        
    Returns:
        Dictionary with classification metrics
    """
    if not label_set:
        label_set = sorted(list(set(gold_labels + predicted_labels)))
            
    # Calculate precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels, 
        predicted_labels, 
        labels=label_set, 
        average=None
    )
    
    # Calculate macro and weighted average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gold_labels, 
        predicted_labels, 
        labels=label_set, 
        average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        gold_labels, 
        predicted_labels, 
        labels=label_set, 
        average='weighted'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(gold_labels, predicted_labels, labels=label_set)
    
    # Format the results
    class_metrics = {}
    for i, label in enumerate(label_set):
        class_metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    metrics = {
        "by_class": class_metrics,
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1)
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1)
        },
        "confusion_matrix": cm.tolist(),
        "labels": label_set
    }
    
    return metrics

def calculate_runtime_metrics(processing_time: float, 
                             text_length: int,
                             entity_count: int,
                             keyword_count: int = None) -> Dict[str, Any]:
    """
    Calculate runtime performance metrics
    
    Args:
        processing_time: Processing time in seconds
        text_length: Length of the processed text in characters
        entity_count: Number of entities detected
        keyword_count: Number of keywords extracted (optional)
        
    Returns:
        Dictionary with runtime metrics
    """
    metrics = {
        "processing_time_ms": processing_time * 1000,
        "text_length": text_length,
        "chars_per_second": text_length / processing_time if processing_time > 0 else 0,
        "entities_per_second": entity_count / processing_time if processing_time > 0 else 0,
        "entity_density": entity_count / text_length if text_length > 0 else 0
    }
    
    if keyword_count is not None:
        metrics.update({
            "keyword_count": keyword_count,
            "keywords_per_second": keyword_count / processing_time if processing_time > 0 else 0,
            "keyword_density": keyword_count / text_length if text_length > 0 else 0
        })
    
    return metrics

def plot_confusion_matrix(cm: List[List[float]], 
                         labels: List[str],
                         output_path: str = None) -> str:
    """
    Plot confusion matrix and return as base64 encoded image or save to file
    
    Args:
        cm: Confusion matrix as nested list
        labels: Labels for classes
        output_path: Optional path to save the plot
        
    Returns:
        Base64 encoded PNG image if output_path is None, otherwise None
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Normalize the confusion matrix
    cm_norm = np.array(cm).astype('float') / np.sum(cm, axis=1)[:, np.newaxis]
    
    # Add text annotations
    thresh = np.array(cm).max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{cm[i][j]}\n({cm_norm[i][j]:.2f})",
                    horizontalalignment="center",
                    color="white" if cm[i][j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return None
    else:
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str

def plot_entity_distribution(entities: List[Dict[str, Any]], 
                            output_path: str = None) -> str:
    """
    Plot entity type distribution and return as base64 encoded image or save to file
    
    Args:
        entities: List of entity dictionaries with 'label' field
        output_path: Optional path to save the plot
        
    Returns:
        Base64 encoded PNG image if output_path is None, otherwise None
    """
    labels = [e.get('label', 'unknown') for e in entities]
    label_counts = Counter(labels)
    
    plt.figure(figsize=(12, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title('Entity Type Distribution')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return None
    else:
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str

def run_entity_extraction_test(gold_standard_file: str, test_texts: List[str]) -> Dict[str, Any]:
    """
    Run entity extraction test against a gold standard file
    
    Args:
        gold_standard_file: Path to JSON file with gold standard entities
        test_texts: List of texts to process and evaluate
        
    Returns:
        Dictionary with test results and metrics
    """
    # Load gold standard data
    try:
        with open(gold_standard_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading gold standard file: {str(e)}")
        return {"error": f"Failed to load gold standard: {str(e)}"}
    
    # Process test texts and extract entities
    start_time = time.time()
    all_predicted_entities = []
    results = []
    for text in test_texts:
        try:
            # Process text with NLP pipeline
            doc = process_text(text)
            
            # Extract entities from the document
            entities = extract_entities_from_doc(doc)
            all_predicted_entities.extend(entities)
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
    
    processing_time = time.time() - start_time
    
    
    for data_case in gold_data:
        answer_entities = data_case.get("entities", [])
        # Calculate entity metrics
        entity_metrics = calculate_entity_metrics(
            gold_entities=answer_entities,
            predicted_entities=all_predicted_entities
        )
        
        # Calculate runtime metrics
        total_text_length = sum(len(text) for text in test_texts)
        runtime_metrics = calculate_runtime_metrics(
            processing_time=processing_time,
            text_length=total_text_length,
            entity_count=len(all_predicted_entities)
        )
        
        # Generate confusion matrix plot if labels are available
        confusion_matrix_plot = None
        if "by_label" in entity_metrics and len(entity_metrics["by_label"]) > 1:
            # Extract labels for classification metrics
            label_set = list(entity_metrics["by_label"].keys())
            gold_labels = [e.get("label") for e in answer_entities]
            pred_labels = [e.get("label") for e in all_predicted_entities]
            
            # Calculate classification metrics for entity labels
            classification_metrics = evaluate_text_classification(
                gold_labels=gold_labels,
                predicted_labels=pred_labels,
                label_set=label_set
            )
            
            # Generate confusion matrix plot
            confusion_matrix_plot = plot_confusion_matrix(
                cm=classification_metrics["confusion_matrix"],
                labels=classification_metrics["labels"]
            )
        
        # Compile full results
        results.append({
            "entity_metrics": entity_metrics,
            "runtime_metrics": runtime_metrics,
            "processed_entity_count": len(all_predicted_entities),
            "gold_entity_count": len(answer_entities),
            "entities": all_predicted_entities,
            "confusion_matrix_plot": confusion_matrix_plot
        })
    
    return results

def run_medical_term_extraction_test(gold_standard_file: str, test_texts: List[str]) -> Dict[str, Any]:
    """
    Run medical term extraction test against a gold standard file
    
    Args:
        gold_standard_file: Path to JSON file with gold standard medical terms
        test_texts: List of texts to process and evaluate
        
    Returns:
        Dictionary with test results and metrics
    """
    # Load gold standard data
    try:
        with open(gold_standard_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading gold standard file: {str(e)}")
        return {"error": f"Failed to load gold standard: {str(e)}"}
    
    # Process test texts and extract medical terms
    start_time = time.time()
    all_predicted_terms = []
    results = []
    for text in test_texts:
        try:
            # Process text with NLP pipeline
            doc = process_text(text)
            
            # Extract medical terms with modifiers
            terms = find_medical_modifiers(doc=doc)
            all_predicted_terms.extend(terms)
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
    
    processing_time = time.time() - start_time
    
    for data_case in gold_data:
        expected_keyweord = data_case.get("expected_keywords", [])
        # Evaluate medical term extraction
        term_metrics = evaluate_medical_term_extraction(
            gold_data=expected_keyweord,
            predicted_data=all_predicted_terms
        )
        
        # Calculate runtime metrics
        total_text_length = sum(len(text) for text in test_texts)
        runtime_metrics = calculate_runtime_metrics(
            processing_time=processing_time,
            text_length=total_text_length,
            entity_count=len(all_predicted_terms),
            keyword_count=sum(len(term.get("modifiers", [])) for term in all_predicted_terms)
        )
        
        # Compile full results
        results.append({
            "term_metrics": term_metrics,
            "runtime_metrics": runtime_metrics,
            "processed_term_count": len(all_predicted_terms),
            "gold_term_count": len(expected_keyweord),
            "terms": all_predicted_terms
        })
    
    return results

def save_test_results(results: Dict[str, Any], output_dir: str, test_name: str) -> None:
    """
    Save test results to files
    
    Args:
        results: Dictionary with test results
        output_dir: Directory to save results
        test_name: Name of the test (used for filenames)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"{test_name}_results.json")
    
    # Create a copy of results without plot data for JSON serialization
    json_results = results.copy()
    if "confusion_matrix_plot" in json_results:
        del json_results["confusion_matrix_plot"]
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    # Save HTML report
    html_path = os.path.join(output_dir, f"{test_name}_report.html")
    generate_metrics_report(results, html_path)
    
    # Save confusion matrix plot if available
    if "confusion_matrix_plot" in results and results["confusion_matrix_plot"]:
        plot_path = os.path.join(output_dir, f"{test_name}_confusion_matrix.png")
        with open(plot_path, 'wb') as f:
            f.write(results["confusion_matrix_plot"].encode('utf-8'))
    
    logger.info(f"Test results saved to {output_dir}")
    
def generate_metrics_report(metrics: Dict[str, Any], 
                          output_path: str,
                          include_plots: bool = True) -> bool:
    """
    Generate HTML report from metrics data
    
    Args:
        metrics: Dictionary with metrics data
        output_path: Path to save the HTML report
        include_plots: Whether to include plots in the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>NLP Pipeline Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric-good {{ color: green; }}
        .metric-average {{ color: orange; }}
        .metric-poor {{ color: red; }}
        .plot-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>NLP Pipeline Evaluation Report</h1>
    <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        # Add entity metrics section if available
        if "entity_metrics" in metrics and "overall" in metrics["entity_metrics"]:
            entity_metrics = metrics["entity_metrics"]
            overall = entity_metrics["overall"]
            counts = entity_metrics.get("counts", {})
            
            html += f"""
    <h2>Entity Extraction Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Precision</td>
            <td class="{'metric-good' if overall.get('precision', 0) > 0.8 else 'metric-average' if overall.get('precision', 0) > 0.5 else 'metric-poor'}">{overall.get('precision', 0):.4f}</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td class="{'metric-good' if overall.get('recall', 0) > 0.8 else 'metric-average' if overall.get('recall', 0) > 0.5 else 'metric-poor'}">{overall.get('recall', 0):.4f}</td>
        </tr>
        <tr>
            <td>F1 Score</td>
            <td class="{'metric-good' if overall.get('f1_score', 0) > 0.8 else 'metric-average' if overall.get('f1_score', 0) > 0.5 else 'metric-poor'}">{overall.get('f1_score', 0):.4f}</td>
        </tr>
        <tr>
            <td>Error Rate</td>
            <td class="{'metric-good' if overall.get('error_rate', 1) < 0.2 else 'metric-average' if overall.get('error_rate', 1) < 0.5 else 'metric-poor'}">{overall.get('error_rate', 0):.4f}</td>
        </tr>
    </table>
    
    <h3>Counts</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>True Positives</td>
            <td>{counts.get('true_positives', 0)}</td>
        </tr>
        <tr>
            <td>False Positives</td>
            <td>{counts.get('false_positives', 0)}</td>
        </tr>
        <tr>
            <td>False Negatives</td>
            <td>{counts.get('false_negatives', 0)}</td>
        </tr>
        <tr>
            <td>Gold Standard Count</td>
            <td>{counts.get('gold_count', 0)}</td>
        </tr>
        <tr>
            <td>Predicted Count</td>
            <td>{counts.get('pred_count', 0)}</td>
        </tr>
    </table>
"""

            # Add per-label metrics if available
            if "by_label" in entity_metrics and entity_metrics["by_label"]:
                html += f"""
    <h3>Metrics by Label</h3>
    <table>
        <tr>
            <th>Label</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Support</th>
        </tr>
"""
                for label, label_metrics in entity_metrics["by_label"].items():
                    html += f"""
        <tr>
            <td>{label}</td>
            <td>{label_metrics.get('precision', 0):.4f}</td>
            <td>{label_metrics.get('recall', 0):.4f}</td>
            <td>{label_metrics.get('f1', 0):.4f}</td>
            <td>{label_metrics.get('support', 0)}</td>
        </tr>
"""
                html += """
    </table>
"""

        # Add classification metrics if available
        if "classification_metrics" in metrics:
            classification_metrics = metrics["classification_metrics"]
            
            html += f"""
    <h2>Classification Metrics</h2>
"""
            
            # Add macro and weighted averages
            if "macro_avg" in classification_metrics and "weighted_avg" in classification_metrics:
                macro_avg = classification_metrics["macro_avg"]
                weighted_avg = classification_metrics["weighted_avg"]
                
                html += f"""
    <h3>Overall Averages</h3>
    <table>
        <tr>
            <th>Type</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
        </tr>
        <tr>
            <td>Macro Average</td>
            <td>{macro_avg.get('precision', 0):.4f}</td>
            <td>{macro_avg.get('recall', 0):.4f}</td>
            <td>{macro_avg.get('f1', 0):.4f}</td>
        </tr>
        <tr>
            <td>Weighted Average</td>
            <td>{weighted_avg.get('precision', 0):.4f}</td>
            <td>{weighted_avg.get('recall', 0):.4f}</td>
            <td>{weighted_avg.get('f1', 0):.4f}</td>
        </tr>
    </table>
"""
            
            # Add per-class metrics
            if "by_class" in classification_metrics and classification_metrics["by_class"]:
                html += f"""
    <h3>Metrics by Class</h3>
    <table>
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Support</th>
        </tr>
"""
                for label, label_metrics in classification_metrics["by_class"].items():
                    html += f"""
        <tr>
            <td>{label}</td>
            <td>{label_metrics.get('precision', 0):.4f}</td>
            <td>{label_metrics.get('recall', 0):.4f}</td>
            <td>{label_metrics.get('f1', 0):.4f}</td>
            <td>{label_metrics.get('support', 0)}</td>
        </tr>
"""
                html += """
    </table>
"""

        # Add runtime metrics if available
        if "runtime_metrics" in metrics:
            runtime_metrics = metrics["runtime_metrics"]
            
            html += f"""
    <h2>Runtime Performance</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Processing Time (ms)</td>
            <td>{runtime_metrics.get('processing_time_ms', 0):.2f}</td>
        </tr>
        <tr>
            <td>Text Length (chars)</td>
            <td>{runtime_metrics.get('text_length', 0)}</td>
        </tr>
        <tr>
            <td>Processing Speed (chars/sec)</td>
            <td>{runtime_metrics.get('chars_per_second', 0):.2f}</td>
        </tr>
"""
            if "entity_density" in runtime_metrics:
                html += f"""
        <tr>
            <td>Entity Density (entities/char)</td>
            <td>{runtime_metrics.get('entity_density', 0):.6f}</td>
        </tr>
"""
            if "keyword_count" in runtime_metrics:
                html += f"""
        <tr>
            <td>Keyword Count</td>
            <td>{runtime_metrics.get('keyword_count', 0)}</td>
        </tr>
        <tr>
            <td>Keywords per Second</td>
            <td>{runtime_metrics.get('keywords_per_second', 0):.2f}</td>
        </tr>
"""
            html += """
    </table>
"""

        # Add confusion matrix plot if available and plots are included
        if include_plots and "confusion_matrix_plot" in metrics and metrics["confusion_matrix_plot"]:
            html += f"""
    <div class="plot-container">
        <h3>Confusion Matrix</h3>
        <img src="data:image/png;base64,{metrics['confusion_matrix_plot']}" alt="Confusion Matrix" />
    </div>
"""

        html += """
</body>
</html>
"""

        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        logger.info(f"Metrics report saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating metrics report: {str(e)}")
        return False

def extract_entities_from_doc(doc) -> List[Dict[str, Any]]:
    """
    Extract entity information from a spaCy Doc object
    
    Args:
        doc: spaCy Doc object
        
    Returns:
        List of entity dictionaries
    """
    entities = []
    for ent in doc.ents:
        is_medical = ent._.is_medical_term
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "is_medical": is_medical
        })
    return entities