# tests/run_tests_metrics.py
#!/usr/bin/env python3
import pytest
import sys
import os
import argparse
from datetime import datetime
from generate_test_summary import generate_summary

def main():
    """Run tests with error rate calculation"""
    parser = argparse.ArgumentParser(description="Run MedSync tests with metrics")
    parser.add_argument("--output-dir", default=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                        help="Directory to save test results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure data directory exists
    os.makedirs("tests/data", exist_ok=True)
    
    # JSON file to collect results
    results_file = "tests/data/test_results.json"
    
    # Clear any existing results
    if os.path.exists(results_file):
        os.remove(results_file)
    
    # Run pytest with JSON output plugin
    pytest_args = [
        "-v",
        "tests/integration/test_error_rates.py",
    ]
    
    print(f"Running tests with command: pytest {' '.join(pytest_args)}")
    return_code = pytest.main(pytest_args)
    
    # Generate summary report if the results exist
    if os.path.exists(results_file):
        generate_summary(results_file, args.output_dir)
    else:
        print(f"No results file found at {results_file}")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())