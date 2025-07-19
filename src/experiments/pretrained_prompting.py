#!/usr/bin/env python3
"""
Pretrained Models Prompting Experiment
====================================

Script này chỉ thực hiện thí nghiệm prompting với pre-trained models:
- meta-llama/Llama-3.1-8B-Instruct
- google/gemma-7b-it

Thực hiện:
1. Zero-shot prompting với pre-trained models
2. Few-shot prompting với pre-trained models
3. Đánh giá và xuất kết quả
"""

import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import os
import random
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PretrainedPromptingExperiment:
    """Experiment class for pretrained models prompting"""
    
    def __init__(self):
        self.models = {
            'meta_llama3': 'meta-llama/Llama-3.1-8B-Instruct',
            'gemma': 'google/gemma-7b-it'
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.tokenizers = {}
        self.loaded_models = {}
        
        self.results = {}
        
    def load_data(self):
        """Load train, validation, and test data"""
        logger.info("Loading data...")
        
        # Load JSON data
        with open('simple_dataset/train/train.json', 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
        
        with open('simple_dataset/val/val.json', 'r', encoding='utf-8') as f:
            self.val_data = json.load(f)
            
        with open('simple_dataset/test/test.json', 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        logger.info(f"Train: {len(self.train_data)} samples")
        logger.info(f"Val: {len(self.val_data)} samples") 
        logger.info(f"Test: {len(self.test_data)} samples")
        
        # Create few-shot examples (6 examples from training data: 3 clickbait, 3 non-clickbait)
        clickbait_examples = [item for item in self.train_data if item['label'] == 'clickbait']
        non_clickbait_examples = [item for item in self.train_data if item['label'] == 'non-clickbait']
        
        self.few_shot_examples = []
        self.few_shot_examples.extend(random.sample(clickbait_examples, 3))
        self.few_shot_examples.extend(random.sample(non_clickbait_examples, 3))
        
        # Shuffle to avoid pattern
        random.shuffle(self.few_shot_examples)
        
        logger.info("Few-shot examples selected:")
        for i, example in enumerate(self.few_shot_examples, 1):
            logger.info(f"  {i}. {example['title'][:50]}... -> {example['label']}")
        
    def setup_model(self, model_name: str, model_path: str):
        """Setup tokenizer and model for a specific model"""
        logger.info(f"Setting up {model_name} ({model_path})...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model with memory optimization
            if 'gemma' in model_name:
                # Gemma with 4-bit quantization to save VRAM
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map='auto',
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map='auto',
                    low_cpu_mem_usage=True
                )
            
            self.tokenizers[model_name] = tokenizer
            self.loaded_models[model_name] = model
            
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            return False
    
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
    
    def generate_prediction(self, model_name: str, prompt: str) -> str:
        """Generate prediction using the model"""
        tokenizer = self.tokenizers[model_name]
        model = self.loaded_models[model_name]
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # Increased from 256 to handle few-shot prompts
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,  # Increased from 5 to 15 for better response generation
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Clear cache after each inference
        torch.cuda.empty_cache()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Debug logging for few-shot issues
        if len(response) < 5:
            logger.debug(f"Short response from {model_name}: '{response}'")
        
        # Extract classification
        if "clickbait" in response.lower() and "non-clickbait" not in response.lower():
            return "clickbait"
        elif "non-clickbait" in response.lower():
            return "non-clickbait"
        else:
            logger.debug(f"Unclear response from {model_name}: '{response}' -> defaulting to non-clickbait")
            return "non-clickbait"  # Default to non-clickbait if unclear
    
    def evaluate_model(self, model_name: str, few_shot: bool = False) -> Dict:
        """Evaluate model on test set"""
        logger.info(f"Evaluating {model_name} ({'few-shot' if few_shot else 'zero-shot'})...")
        
        predictions = []
        true_labels = []
        
        start_time = time.time()
        
        for i, item in enumerate(self.test_data):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                remaining = len(self.test_data) - (i + 1)
                eta = elapsed * remaining / (i + 1)
                logger.info(f"Processed {i+1}/{len(self.test_data)} samples. ETA: {eta/60:.1f} minutes")
            
            prompt = self.create_prompt(item['title'], few_shot=few_shot)
            pred = self.generate_prediction(model_name, prompt)
            predictions.append(pred)
            true_labels.append(item['label'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # For binary classification
        clickbait_f1 = f1_score(true_labels, predictions, pos_label='clickbait')
        clickbait_precision = precision_score(true_labels, predictions, pos_label='clickbait')
        
        non_clickbait_f1 = f1_score(true_labels, predictions, pos_label='non-clickbait')
        non_clickbait_precision = precision_score(true_labels, predictions, pos_label='non-clickbait')
        
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'clickbait_f1': clickbait_f1,
            'clickbait_precision': clickbait_precision,
            'non_clickbait_f1': non_clickbait_f1,
            'non_clickbait_precision': non_clickbait_precision,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        logger.info(f"Results for {model_name} ({'few-shot' if few_shot else 'zero-shot'}):")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Macro F1: {macro_f1:.4f}")
        logger.info(f"  Weighted F1: {weighted_f1:.4f}")
        logger.info(f"  Clickbait F1: {clickbait_f1:.4f}")
        logger.info(f"  Non-clickbait F1: {non_clickbait_f1:.4f}")
        
        return results
    
    def save_results(self, results: Dict, filename: str):
        """Save results to CSV file"""
        logger.info(f"Saving results to {filename}...")
        
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
        df.to_csv(filename, index=False, encoding='utf-8')
        
        logger.info(f"Results saved to {filename}")
    
    def run_pretrained_experiments(self):
        """Run experiments with pretrained models only - one model at a time to save VRAM"""
        logger.info("Starting pretrained models prompting experiments...")
        
        # Load data
        self.load_data()
        
        # Run experiments with each model individually to optimize VRAM
        logger.info("\n" + "="*60)
        logger.info("PRETRAINED MODELS PROMPTING EXPERIMENTS")
        logger.info("="*60)
        
        for model_name, model_path in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"LOADING AND TESTING {model_name.upper()}")
            logger.info(f"{'='*50}")
            
            # Load model
            success = self.setup_model(model_name, model_path)
            if not success:
                logger.warning(f"✗ Skipping {model_name} due to loading error")
                continue
            
            logger.info(f"✓ {model_name} loaded successfully")
            
            if model_name not in self.results:
                self.results[model_name] = {}
            
            # Zero-shot
            logger.info(f"\n>>> Zero-shot prompting for {model_name}")
            self.results[model_name]['pretrained_zeroshot'] = self.evaluate_model(
                model_name, few_shot=False
            )
            
            # Clear GPU memory after zero-shot
            torch.cuda.empty_cache()
            logger.info(f"Memory cleared after zero-shot for {model_name}")
            
            # Few-shot
            logger.info(f"\n>>> Few-shot prompting for {model_name}")
            self.results[model_name]['pretrained_fewshot'] = self.evaluate_model(
                model_name, few_shot=True
            )
            
            # Unload model to free VRAM for next model
            del self.loaded_models[model_name]
            del self.tokenizers[model_name]
            torch.cuda.empty_cache()
            logger.info(f"✓ Completed and unloaded {model_name}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pretrained_prompting_results_{timestamp}.csv"
        self.save_results(self.results, filename)
        
        logger.info(f"\n{'='*60}")
        logger.info("PRETRAINED PROMPTING EXPERIMENTS COMPLETED!")
        logger.info(f"{'='*60}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all results"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)
        
        for model_name, model_results in self.results.items():
            logger.info(f"\n{model_name.upper()}:")
            for experiment_type, metrics in model_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    logger.info(f"  {experiment_type:25}: "
                               f"Acc={metrics['accuracy']:.4f}, "
                               f"Macro F1={metrics['macro_f1']:.4f}, "
                               f"Weighted F1={metrics['weighted_f1']:.4f}")
        
        logger.info(f"\n{'='*60}")
        logger.info("READY FOR FINE-TUNING EXPERIMENTS!")
        logger.info("="*60)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run pretrained experiments
    experiment = PretrainedPromptingExperiment()
    experiment.run_pretrained_experiments() 