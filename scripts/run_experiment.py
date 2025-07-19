#!/usr/bin/env python3
"""
Simple Test Script for Clickbait Detection
==========================================

This script runs a basic test with one model to verify the setup.
"""

import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleClickbaitTest:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
    def load_data(self):
        """Load test data"""
        logger.info("Loading test data...")
        
        with open('data/simple_dataset/test/test.json', 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        with open('data/simple_dataset/train/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Create balanced few-shot examples (3 clickbait, 3 non-clickbait)
        clickbait_examples = [item for item in train_data if item['label'] == 'clickbait']
        non_clickbait_examples = [item for item in train_data if item['label'] == 'non-clickbait']
        
        self.few_shot_examples = []
        self.few_shot_examples.extend(random.sample(clickbait_examples, 3))
        self.few_shot_examples.extend(random.sample(non_clickbait_examples, 3))
        
        # Shuffle to avoid pattern
        random.shuffle(self.few_shot_examples)
        
        logger.info(f"Test: {len(self.test_data)} samples")
        
    def create_prompt(self, title: str, few_shot: bool = False) -> str:
        """Create prompt for clickbait detection"""
        
        system_prompt = """Bạn là một chuyên gia phân tích nội dung. Nhiệm vụ của bạn là phân loại tiêu đề tin tức thành "clickbait" hoặc "non-clickbait".

Clickbait là những tiêu đề được thiết kế để thu hút người đọc nhấp chuột bằng cách sử dụng các từ ngữ gây tò mò, cảm xúc mạnh, hoặc hứa hẹn thông tin mà không tiết lộ trực tiếp.

Trả lời chỉ với "clickbait" hoặc "non-clickbait"."""

        if few_shot:
            examples = ""
            for example in self.few_shot_examples:
                examples += f"Tiêu đề: {example['title']}\nPhân loại: {example['label']}\n\n"
            
            prompt = f"{system_prompt}\n\nMột số ví dụ:\n{examples}Tiêu đề: {title}\nPhân loại:"
        else:
            prompt = f"{system_prompt}\n\nTiêu đề: {title}\nPhân loại:"
            
        return prompt
    
    def test_model(self, model_name: str, num_samples: int = 10):
        """Test a single model with limited samples"""
        logger.info(f"Testing {model_name}...")
        
        try:
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True
            )
            
            logger.info(f"Successfully loaded {model_name}")
            
            # Test with limited samples
            test_samples = self.test_data[:num_samples]
            predictions = []
            true_labels = []
            
            for i, item in enumerate(test_samples):
                logger.info(f"Processing sample {i+1}/{num_samples}...")
                
                # Zero-shot
                prompt = self.create_prompt(item['title'], few_shot=False)
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                # Extract classification
                if "clickbait" in response.lower() and "non-clickbait" not in response.lower():
                    pred = "clickbait"
                elif "non-clickbait" in response.lower():
                    pred = "non-clickbait"
                else:
                    pred = "non-clickbait"
                
                predictions.append(pred)
                true_labels.append(item['label'])
                
                logger.info(f"Title: {item['title'][:50]}...")
                logger.info(f"True: {item['label']}, Pred: {pred}")
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            macro_f1 = f1_score(true_labels, predictions, average='macro')
            
            logger.info(f"Results: Accuracy={accuracy:.4f}, Macro F1={macro_f1:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
            return False

def main():
    """Main function"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run test
    test = SimpleClickbaitTest()
    test.load_data()
    
    # Test with selected models
    models_to_test = [
        'google/gemma-7b-it',  # Try Gemma first as it's usually more stable
        'meta-llama/Llama-3.1-8B-Instruct'
    ]
    
    for model_name in models_to_test:
        logger.info(f"\n=== Testing {model_name} ===")
        success = test.test_model(model_name, num_samples=5)
        if success:
            logger.info(f"✓ {model_name} works correctly")
        else:
            logger.error(f"✗ {model_name} failed")
        
        # Clear GPU memory
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 