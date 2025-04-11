import argparse
import logging
import sys
import os
import json
import time
from typing import List, Dict, Any
from datetime import datetime

# Import the test functions (make sure these match what's in test_nlp_pipeline.py)
from tests.nlp.metrics import (
    run_entity_extraction_test,
    run_medical_term_extraction_test,
    save_test_results
)

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_test_data(file_path: str) -> List[str]:
    """
    Load test texts from a file
    
    Args:
        file_path: Path to text file with test data (one text per line) or JSON file
        
    Returns:
        List of test texts
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")
        
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('texts', [])
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def main() -> None:
    """Main function to run the test suite"""
    parser = argparse.ArgumentParser(description='NLP Pipeline Test Suite')
    parser.add_argument('--test-data', default=os.path.join("tests", "data", "test.txt"), help='Path to test data file (one text per line or JSON)')
    parser.add_argument('--gold-standard', default=os.path.join("tests", "data", "standard.json"), help='Path to gold standard data file (JSON)')
    parser.add_argument('--output-dir', default=os.path.join('tests', 'test_results'), help='Directory to save test results')
    parser.add_argument('--test-type', choices=['entity', 'term', 'classification', 'summarization', 'all'], 
                       default='all', help='Type of test to run')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Path to log file')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    try:
        test_texts = load_test_data(args.test_data)
        logger.info(f"Loaded {len(test_texts)} test texts")
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        sys.exit(1)
    
    # Create timestamp for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Record test configuration
    config = {
        'timestamp': timestamp,
        'test_data': args.test_data,
        'gold_standard': args.gold_standard,
        'test_type': args.test_type,
        'num_texts': len(test_texts)
    }
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run selected tests
    if args.test_type in ['entity', 'all']:
        logger.info("Running entity extraction test...")
        if args.gold_standard:
            entity_results = run_entity_extraction_test(args.gold_standard, test_texts)
            save_test_results(entity_results, run_dir, "entity_extraction")
        else:
            logger.warning("Skipping entity extraction test: No gold standard file provided")
    
    if args.test_type in ['term', 'all']:
        logger.info("Running medical term extraction test...")
        if args.gold_standard:
            term_results = run_medical_term_extraction_test(args.gold_standard, test_texts)
            save_test_results(term_results, run_dir, "term_extraction")
        else:
            logger.warning("Skipping medical term extraction test: No gold standard file provided")
    
    logger.info(f"All tests completed. Results saved to '{run_dir}'")

if __name__ == "__main__":
    main()