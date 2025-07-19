"""
Data loading utilities for clickbait detection experiments
"""

import json
import pandas as pd
from typing import List, Dict, Tuple
import random
import logging

logger = logging.getLogger(__name__)

class ClickbaitDataLoader:
    """Utility class for loading and preprocessing clickbait detection data"""
    
    def __init__(self, data_dir: str = "data/simple_dataset"):
        self.data_dir = data_dir
        
    def load_json_data(self, split: str) -> List[Dict]:
        """Load data from JSON file"""
        file_path = f"{self.data_dir}/{split}/{split}.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
    
    def load_csv_data(self, split: str) -> pd.DataFrame:
        """Load data from CSV file"""
        file_path = f"{self.data_dir}/{split}/{split}.csv"
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
    
    def get_few_shot_examples(self, train_data: List[Dict], 
                             n_clickbait: int = 3, 
                             n_non_clickbait: int = 3,
                             shuffle: bool = True) -> List[Dict]:
        """Get balanced few-shot examples"""
        clickbait_examples = [item for item in train_data if item['label'] == 'clickbait']
        non_clickbait_examples = [item for item in train_data if item['label'] == 'non-clickbait']
        
        few_shot_examples = []
        if len(clickbait_examples) >= n_clickbait:
            few_shot_examples.extend(random.sample(clickbait_examples, n_clickbait))
        else:
            few_shot_examples.extend(clickbait_examples)
            
        if len(non_clickbait_examples) >= n_non_clickbait:
            few_shot_examples.extend(random.sample(non_clickbait_examples, n_non_clickbait))
        else:
            few_shot_examples.extend(non_clickbait_examples)
        
        if shuffle:
            random.shuffle(few_shot_examples)
            
        return few_shot_examples
    
    def get_data_statistics(self, data: List[Dict]) -> Dict:
        """Get basic statistics about the data"""
        if not data:
            return {}
            
        total = len(data)
        clickbait_count = sum(1 for item in data if item['label'] == 'clickbait')
        non_clickbait_count = total - clickbait_count
        
        return {
            'total_samples': total,
            'clickbait_samples': clickbait_count,
            'non_clickbait_samples': non_clickbait_count,
            'clickbait_ratio': clickbait_count / total if total > 0 else 0,
            'non_clickbait_ratio': non_clickbait_count / total if total > 0 else 0
        }
