# tests/integration/test_error_rates.py
import json
import pytest
import os

# Load gold standard test data
try:
    with open('tests/data/standard.json', 'r') as f:
        test_data = json.load(f)
except FileNotFoundError:
    # Fallback if file not at expected location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'standard.json')
    with open(data_path, 'r') as f:
        test_data = json.load(f)

@pytest.mark.parametrize("case", test_data)
def test_error_rates(client, case):
    """Test error rates against gold standard data"""
    transcript = case['transcript']
    gold_entities = case['entities']
    
    # Process the transcript
    response = client.post(
        "/tests/basic-test",
        json={"text": transcript}
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Get entities from the response
    predicted_entities = result.get('entities', [])
    
    # Calculate basic metrics
    gold_entities_texts = {entity['text'].lower() for entity in gold_entities}
    pred_entities_texts = {entity['text'].lower() for entity in predicted_entities}
    
    # Calculate intersection (true positives)
    true_positives = len(gold_entities_texts.intersection(pred_entities_texts))
    
    # Calculate precision and recall
    precision = true_positives / len(pred_entities_texts) if pred_entities_texts else 0
    recall = true_positives / len(gold_entities_texts) if gold_entities_texts else 0
    
    # Calculate F1 score and error rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = 1.0 - f1
    
    # Print metrics for this case
    print(f"\nCase ID: {case.get('case_id', 'unknown')}")
    print(f"Entity Recognition - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    print(f"Entity Recognition - Error Rate: {error_rate:.2f}")
    
    # Save the results in a structured way to a JSON file
    os.makedirs("tests/data", exist_ok=True)
    result_file = "tests/data/test_results.json"
    
    # Create the result object
    result_obj = {
        "case_id": case.get('case_id', 'unknown'),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_rate": error_rate
        }
    }
    
    # Save the result
    try:
        existing_results = []
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = []
    except:
        existing_results = []
    
    existing_results.append(result_obj)
    
    with open(result_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Assert error rate is below acceptable threshold
    max_error_rate = 0.6  # 60% error rate allowed for now
    assert error_rate <= max_error_rate, f"Entity recognition error rate too high: {error_rate:.2f}"