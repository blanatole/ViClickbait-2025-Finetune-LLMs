"""
Evaluation metrics for clickbait detection
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
import pandas as pd
from typing import List, Dict, Any
import numpy as np

class ClickbaitMetrics:
    """Utility class for calculating evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate comprehensive metrics for binary classification"""
        
        # Convert to binary format for sklearn
        y_true_binary = [1 if label == 'clickbait' else 0 for label in y_true]
        y_pred_binary = [1 if label == 'clickbait' else 0 for label in y_pred]
        
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision_macro': precision_score(y_true_binary, y_pred_binary, average='macro'),
            'recall_macro': recall_score(y_true_binary, y_pred_binary, average='macro'),
            'f1_macro': f1_score(y_true_binary, y_pred_binary, average='macro'),
            'precision_clickbait': precision_score(y_true_binary, y_pred_binary, pos_label=1),
            'recall_clickbait': recall_score(y_true_binary, y_pred_binary, pos_label=1),
            'f1_clickbait': f1_score(y_true_binary, y_pred_binary, pos_label=1),
            'precision_non_clickbait': precision_score(y_true_binary, y_pred_binary, pos_label=0),
            'recall_non_clickbait': recall_score(y_true_binary, y_pred_binary, pos_label=0),
            'f1_non_clickbait': f1_score(y_true_binary, y_pred_binary, pos_label=0)
        }
        
        return metrics
    
    @staticmethod
    def print_detailed_report(y_true: List[str], y_pred: List[str], 
                            model_name: str = "Model") -> None:
        """Print detailed classification report"""
        
        y_true_binary = [1 if label == 'clickbait' else 0 for label in y_true]
        y_pred_binary = [1 if label == 'clickbait' else 0 for label in y_pred]
        
        print(f"\n=== {model_name} Evaluation Results ===")
        print(f"Total samples: {len(y_true)}")
        print(f"Accuracy: {accuracy_score(y_true_binary, y_pred_binary):.4f}")
        
        print("\nDetailed Classification Report:")
        target_names = ['non-clickbait', 'clickbait']
        print(classification_report(y_true_binary, y_pred_binary, 
                                  target_names=target_names, digits=4))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        print(f"                Predicted")
        print(f"                Non-CB  Clickbait")
        print(f"Actual Non-CB   {cm[0,0]:6d}  {cm[0,1]:9d}")
        print(f"       Clickbait{cm[1,0]:6d}  {cm[1,1]:9d}")
    
    @staticmethod
    def save_results_to_csv(results: List[Dict[str, Any]], 
                           filename: str) -> None:
        """Save experiment results to CSV file"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    @staticmethod
    def compare_models(results: List[Dict[str, Any]], 
                      metric: str = 'f1_macro') -> pd.DataFrame:
        """Compare multiple models by a specific metric"""
        df = pd.DataFrame(results)
        if metric in df.columns:
            df_sorted = df.sort_values(by=metric, ascending=False)
            return df_sorted[['model_name', 'experiment_type', metric]]
        else:
            print(f"Metric '{metric}' not found in results")
            return pd.DataFrame()
