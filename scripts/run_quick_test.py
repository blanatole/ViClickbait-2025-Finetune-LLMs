#!/usr/bin/env python3
"""
Quick Test Script for Clickbait Detection Setup
=============================================

This script tests the basic functionality without downloading large models.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading functionality"""
    logger.info("Testing data loading...")
    
    try:
        # Load data
        with open('simple_dataset/train/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open('simple_dataset/val/val.json', 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            
        with open('simple_dataset/test/test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"Train: {len(train_data)} samples")
        logger.info(f"Val: {len(val_data)} samples") 
        logger.info(f"Test: {len(test_data)} samples")
        
        # Check data format
        logger.info("Sample data:")
        for i, sample in enumerate(train_data[:3]):
            logger.info(f"  {i+1}. Title: {sample['title'][:50]}...")
            logger.info(f"     Label: {sample['label']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

def test_prompt_creation():
    """Test prompt creation"""
    logger.info("Testing prompt creation...")
    
    try:
        # Load sample data
        with open('simple_dataset/train/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Create balanced few-shot examples (3 clickbait, 3 non-clickbait)
        clickbait_examples = [item for item in train_data if item['label'] == 'clickbait']
        non_clickbait_examples = [item for item in train_data if item['label'] == 'non-clickbait']
        
        few_shot_examples = []
        few_shot_examples.extend(random.sample(clickbait_examples, 3))
        few_shot_examples.extend(random.sample(non_clickbait_examples, 3))
        
        # Shuffle to avoid pattern
        random.shuffle(few_shot_examples)
        
        # Test zero-shot prompt
        title = "Bí mật khủng khiếp đằng sau nụ cười của mỹ nhân này"
        
        system_prompt = """Bạn là một chuyên gia phân tích nội dung. Nhiệm vụ của bạn là phân loại tiêu đề tin tức thành "clickbait" hoặc "non-clickbait".

Clickbait là những tiêu đề được thiết kế để thu hút người đọc nhấp chuột bằng cách sử dụng các từ ngữ gây tò mò, cảm xúc mạnh, hoặc hứa hẹn thông tin mà không tiết lộ trực tiếp.

Trả lời chỉ với "clickbait" hoặc "non-clickbait"."""
        
        # Zero-shot prompt
        zero_shot_prompt = f"{system_prompt}\n\nTiêu đề: {title}\nPhân loại:"
        
        # Few-shot prompt
        examples = ""
        for example in few_shot_examples:
            examples += f"Tiêu đề: {example['title']}\nPhân loại: {example['label']}\n\n"
        
        few_shot_prompt = f"{system_prompt}\n\nMột số ví dụ:\n{examples}Tiêu đề: {title}\nPhân loại:"
        
        logger.info("Zero-shot prompt created successfully")
        logger.info("Few-shot prompt created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating prompts: {str(e)}")
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    logger.info("Testing metrics calculation...")
    
    try:
        # Create dummy predictions
        true_labels = ['clickbait', 'non-clickbait', 'clickbait', 'non-clickbait', 'clickbait']
        predictions = ['clickbait', 'non-clickbait', 'non-clickbait', 'non-clickbait', 'clickbait']
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        clickbait_f1 = f1_score(true_labels, predictions, pos_label='clickbait')
        clickbait_precision = precision_score(true_labels, predictions, pos_label='clickbait')
        
        non_clickbait_f1 = f1_score(true_labels, predictions, pos_label='non-clickbait')
        non_clickbait_precision = precision_score(true_labels, predictions, pos_label='non-clickbait')
        
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Clickbait F1: {clickbait_f1:.4f}")
        logger.info(f"Non-clickbait F1: {non_clickbait_f1:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        logger.info(f"Weighted F1: {weighted_f1:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return False

def test_csv_export():
    """Test CSV export functionality"""
    logger.info("Testing CSV export...")
    
    try:
        # Create dummy results
        results = {
            'vtsnlp_llama3': {
                'pretrained_zeroshot': {
                    'accuracy': 0.8,
                    'clickbait_f1': 0.75,
                    'clickbait_precision': 0.8,
                    'non_clickbait_f1': 0.85,
                    'non_clickbait_precision': 0.8,
                    'macro_f1': 0.8,
                    'weighted_f1': 0.8
                }
            }
        }
        
        # Convert to DataFrame
        rows = []
        for model_name, model_results in results.items():
            for experiment_type, metrics in model_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    row = {
                        'model': model_name,
                        'experiment': experiment_type,
                        'accuracy': metrics['accuracy'],
                        'clickbait_f1': metrics['clickbait_f1'],
                        'clickbait_precision': metrics['clickbait_precision'],
                        'non_clickbait_f1': metrics['non_clickbait_f1'],
                        'non_clickbait_precision': metrics['non_clickbait_precision'],
                        'macro_f1': metrics['macro_f1'],
                        'weighted_f1': metrics['weighted_f1']
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv('test_results.csv', index=False, encoding='utf-8')
        
        logger.info("CSV export successful")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("=== Running Quick Setup Test ===")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Prompt Creation", test_prompt_creation),
        ("Metrics Calculation", test_metrics_calculation),
        ("CSV Export", test_csv_export)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {str(e)}")
            failed += 1
    
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Ready to run full experiment.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before running full experiment.")
        return False

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    
    success = main()
    exit(0 if success else 1) 