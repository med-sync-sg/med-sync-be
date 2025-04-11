#!/usr/bin/env python3
# run_tests.py
import pytest
import argparse
import os
import sys

def main():
    
    
    parser = argparse.ArgumentParser(description="Run MedSync tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    args = parser.parse_args()
    
    # Set up test command
    cmd = ["pytest", "-v"]
    
    # Add coverage if requested
    if args.coverage:
        cmd = ["pytest", "--cov=app", "--cov-report=html", "--cov-report=term"]
    
    # Select test type
    if args.unit:
        cmd.append("tests/unit/")
    elif args.integration:
        cmd.append("tests/integration/")
    else:
        # Run all tests by default
        cmd.append("tests/")
    
    # Run tests
    return pytest.main(cmd)

if __name__ == "__main__":
    sys.exit(main())