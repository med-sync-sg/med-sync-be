# tests/generate_test_summary.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def generate_summary(results_file, output_dir):
    """Generate a summary report from test results"""
    try:
        with open(results_file, 'r') as f:
            content = f.read()
            try:
                # Try to parse as a list of objects
                results = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: try parsing each line separately
                results = []
                for line in content.strip().split('\n'):
                    try:
                        results.append(json.loads(line))
                    except:
                        pass
        
        # Ensure we have a list
        if not isinstance(results, list):
            results = [results]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract test metrics
        case_ids = []
        precisions = []
        recalls = []
        f1_scores = []
        error_rates = []
        
        for result in results:
            # Extract case ID
            case_id = result.get('case_id', 'unknown')
            case_ids.append(case_id)
            
            # Extract metrics
            metrics = result.get('metrics', {})
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1_score', 0))
            error_rates.append(metrics.get('error_rate', 1.0))
        
        # Calculate average metrics
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        avg_error = sum(error_rates) / len(error_rates) if error_rates else 1.0
        
        # Create bar chart of error rates
        plt.figure(figsize=(10, 6))
        bars = plt.bar(case_ids, error_rates, color='darkred')
        plt.axhline(y=0.4, color='r', linestyle='--', label='40% Error Threshold')
        
        # Add values on top of bars
        for bar, error in zip(bars, error_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{error:.2f}', ha='center', va='bottom')
        
        plt.xlabel('Test Case')
        plt.ylabel('Error Rate')
        plt.title('Entity Recognition Error Rates by Case')
        plt.ylim(0, 1.1)  # Set y-axis from 0 to 1.1 to leave room for text
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(output_dir, 'error_rates.png'))
        
        # Create summary HTML report
        with open(os.path.join(output_dir, 'summary.html'), 'w') as f:
            f.write(f'''
            <html>
            <head>
                <title>MedSync NLP Test Summary</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .good {{ color: green; }}
                    .warn {{ color: orange; }}
                    .bad {{ color: red; }}
                    img {{ max-width: 100%; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>MedSync NLP Test Summary</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Overall Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>Average Precision</td>
                        <td>{avg_precision:.4f}</td>
                        <td class="{get_status_class(avg_precision)}">{get_status(avg_precision)}</td>
                    </tr>
                    <tr>
                        <td>Average Recall</td>
                        <td>{avg_recall:.4f}</td>
                        <td class="{get_status_class(avg_recall)}">{get_status(avg_recall)}</td>
                    </tr>
                    <tr>
                        <td>Average F1 Score</td>
                        <td>{avg_f1:.4f}</td>
                        <td class="{get_status_class(avg_f1)}">{get_status(avg_f1)}</td>
                    </tr>
                    <tr>
                        <td>Average Error Rate</td>
                        <td>{avg_error:.4f}</td>
                        <td class="{get_error_status_class(avg_error)}">{get_error_status(avg_error)}</td>
                    </tr>
                </table>
                
                <h2>Error Rates by Case</h2>
                <img src="error_rates.png" alt="Error Rates Chart">
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Case ID</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Error Rate</th>
                    </tr>
            ''')
            
            # Add rows for each test case
            for i in range(len(case_ids)):
                f.write(f'''
                    <tr>
                        <td>{case_ids[i]}</td>
                        <td>{precisions[i]:.4f}</td>
                        <td>{recalls[i]:.4f}</td>
                        <td>{f1_scores[i]:.4f}</td>
                        <td class="{get_error_status_class(error_rates[i])}">{error_rates[i]:.4f}</td>
                    </tr>
                ''')
            
            f.write('''
                </table>
            </body>
            </html>
            ''')
            
        print(f"Summary report generated in {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def get_status_class(value):
    """Get CSS class for status based on value"""
    if value >= 0.7:
        return "good"
    elif value >= 0.4:
        return "warn"
    else:
        return "bad"

def get_status(value):
    """Get status text based on value"""
    if value >= 0.7:
        return "Good"
    elif value >= 0.4:
        return "Acceptable"
    else:
        return "Poor"

def get_error_status_class(value):
    """Get CSS class for error status based on value"""
    if value <= 0.3:
        return "good"
    elif value <= 0.6:
        return "warn"
    else:
        return "bad"

def get_error_status(value):
    """Get error status text based on value"""
    if value <= 0.3:
        return "Good"
    elif value <= 0.6:
        return "Acceptable"
    else:
        return "Poor"

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_test_summary.py <results_file> <output_dir>")
        sys.exit(1)
    
    generate_summary(sys.argv[1], sys.argv[2])