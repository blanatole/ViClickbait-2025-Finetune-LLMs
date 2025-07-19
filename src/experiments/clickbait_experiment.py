#!/usr/bin/env python3
"""
Clickbait Detection Experiment Script
====================================

This script implements a comprehensive experiment pipeline for clickbait detection using multiple models:
- meta-llama/Llama-3.1-8B-Instruct
- google/gemma-7b-it

The script performs:
1. Zero-shot and few-shot prompting with pre-trained models
2. Fine-tuning of all models
3. Zero-shot and few-shot prompting with fine-tuned models
4. Comprehensive evaluation with multiple metrics
"""

import torch
import json
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
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

class ClickbaitExperiment:
    """Main experiment class for clickbait detection"""
    
    def __init__(self):
        self.models = {
            'meta_llama3': 'meta-llama/Llama-3.1-8B-Instruct',
            'gemma': 'google/gemma-7b-it'
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.tokenizers = {}
        self.loaded_models = {}
        self.finetuned_models = {}
        
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
        
    def setup_model(self, model_name: str, model_path: str):
        """Setup tokenizer and model for a specific model"""
        logger.info(f"Setting up {model_name} ({model_path})...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model
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
        
        # Use fine-tuned model if available, otherwise use pre-trained
        if model_name in self.finetuned_models:
            model = self.finetuned_models[model_name]
        else:
            model = self.loaded_models[model_name]
            
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
            return "clickbait"
        elif "non-clickbait" in response.lower():
            return "non-clickbait"
        else:
            return "non-clickbait"  # Default to non-clickbait if unclear
    
    def evaluate_model(self, model_name: str, few_shot: bool = False, use_finetuned: bool = False) -> Dict:
        """Evaluate model on test set"""
        logger.info(f"Evaluating {model_name} ({'few-shot' if few_shot else 'zero-shot'}) "
                   f"({'fine-tuned' if use_finetuned else 'pre-trained'})...")
        
        predictions = []
        true_labels = []
        
        for item in self.test_data:
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
        
        logger.info(f"Results for {model_name}: Accuracy={accuracy:.4f}, Macro F1={macro_f1:.4f}")
        
        return results
    
    def prepare_finetuning_data(self):
        """Prepare data for fine-tuning"""
        logger.info("Preparing fine-tuning data...")
        
        finetuning_data = []
        
        for item in self.train_data:
            prompt = self.create_prompt(item['title'], few_shot=False)
            completion = item['label']
            
            # Create input-output pair
            text = f"{prompt} {completion}"
            finetuning_data.append({'text': text})
        
        return Dataset.from_list(finetuning_data)
    
    def finetune_model(self, model_name: str):
        """Fine-tune a specific model"""
        logger.info(f"Fine-tuning {model_name}...")
        
        if model_name not in self.loaded_models:
            logger.error(f"Model {model_name} not loaded. Skipping fine-tuning.")
            return False
            
        try:
            model = self.loaded_models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Prepare dataset
            dataset = self.prepare_finetuning_data()
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./finetuned_{model_name}",
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                warmup_steps=100,
                logging_steps=50,
                save_steps=500,
                evaluation_strategy="no",
                learning_rate=5e-5,
                fp16=True,
                dataloader_num_workers=4,
                remove_unused_columns=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Train
            trainer.train()
            
            # Save fine-tuned model
            trainer.save_model(f"./finetuned_{model_name}")
            
            # Store fine-tuned model
            self.finetuned_models[model_name] = model
            
            logger.info(f"Successfully fine-tuned {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error fine-tuning {model_name}: {str(e)}")
            return False
    
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
    
    def run_all_experiments(self):
        """Run all experiments"""
        logger.info("Starting comprehensive clickbait detection experiments...")
        
        # Load data
        self.load_data()
        
        # Setup all models
        for model_name, model_path in self.models.items():
            success = self.setup_model(model_name, model_path)
            if not success:
                logger.warning(f"Skipping {model_name} due to loading error")
                continue
        
        # Run experiments with pre-trained models
        logger.info("=== Pre-trained Models Experiments ===")
        for model_name in self.loaded_models.keys():
            logger.info(f"\n--- {model_name} ---")
            
            if model_name not in self.results:
                self.results[model_name] = {}
            
            # Zero-shot
            self.results[model_name]['pretrained_zeroshot'] = self.evaluate_model(
                model_name, few_shot=False, use_finetuned=False
            )
            
            # Few-shot
            self.results[model_name]['pretrained_fewshot'] = self.evaluate_model(
                model_name, few_shot=True, use_finetuned=False
            )
        
        # Fine-tune all models
        logger.info("=== Fine-tuning Models ===")
        for model_name in self.loaded_models.keys():
            self.finetune_model(model_name)
        
        # Run experiments with fine-tuned models
        logger.info("=== Fine-tuned Models Experiments ===")
        for model_name in self.finetuned_models.keys():
            logger.info(f"\n--- {model_name} (Fine-tuned) ---")
            
            # Zero-shot with fine-tuned model
            self.results[model_name]['finetuned_zeroshot'] = self.evaluate_model(
                model_name, few_shot=False, use_finetuned=True
            )
            
            # Few-shot with fine-tuned model
            self.results[model_name]['finetuned_fewshot'] = self.evaluate_model(
                model_name, few_shot=True, use_finetuned=True
            )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results(self.results, f"clickbait_results_{timestamp}.csv")
        
        logger.info("All experiments completed!")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all results"""
        logger.info("\n=== EXPERIMENT SUMMARY ===")
        
        for model_name, model_results in self.results.items():
            logger.info(f"\n{model_name.upper()}:")
            for experiment_type, metrics in model_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    logger.info(f"  {experiment_type}: "
                               f"Acc={metrics['accuracy']:.4f}, "
                               f"Macro F1={metrics['macro_f1']:.4f}, "
                               f"Weighted F1={metrics['weighted_f1']:.4f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run experiments
    experiment = ClickbaitExperiment()
    experiment.run_all_experiments() 