#!/usr/bin/env python3
"""
Fine-tuned Models Prompting Evaluation
====================================

Script để đánh giá fine-tuned models bằng prompting:
- Load LoRA fine-tuned models
- Zero-shot và few-shot prompting
- So sánh với pretrained baseline
"""

import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
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

class FineTunedPromptingEvaluation:
    """Evaluation class for fine-tuned models using prompting"""
    
    def __init__(self):
        self.models = {
            'meta_llama3': 'meta-llama/Llama-3.1-8B-Instruct',
            'gemma': 'google/gemma-7b-it'
        }
        
        # Paths to fine-tuned adapters
        self.adapter_paths = {
            'meta_llama3': './qlora_adapter_meta_llama3',
            'gemma': './qlora_adapter_gemma'
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.tokenizers = {}
        self.base_models = {}
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
        
        # Create few-shot examples (same as used in training)
        self.create_few_shot_examples()
        
    def create_few_shot_examples(self):
        """Create few-shot examples with strategic selection for better performance"""
        clickbait_examples = [item for item in self.train_data if item['label'] == 'clickbait']
        non_clickbait_examples = [item for item in self.train_data if item['label'] == 'non-clickbait']
        
        # Strategic selection: diverse examples with different patterns
        # Sort by length to get diverse examples
        clickbait_examples.sort(key=lambda x: len(x['title']))
        non_clickbait_examples.sort(key=lambda x: len(x['title']))
        
        # Select diverse examples: short, medium, long, and some with specific patterns
        selected_clickbait = []
        selected_non_clickbait = []
        
        # For clickbait: pick examples with common clickbait patterns
        clickbait_patterns = [
            lambda x: any(word in x['title'].lower() for word in ['bí mật', 'tuyệt đối', 'không thể', 'sốc']),
            lambda x: '?' in x['title'],
            lambda x: any(word in x['title'].lower() for word in ['top', 'nhất', 'khủng', 'lần đầu']),
            lambda x: any(word in x['title'].lower() for word in ['bất ngờ', 'không ngờ', 'chưa ai']),
            lambda x: len(x['title']) < 50,  # Short titles
            lambda x: len(x['title']) > 80   # Long titles
        ]
        
        for pattern in clickbait_patterns:
            matching = [x for x in clickbait_examples if pattern(x)]
            if matching and len(selected_clickbait) < 5:
                selected_clickbait.append(matching[0])
        
        # Fill remaining slots with diverse examples
        while len(selected_clickbait) < 5:
            for ex in clickbait_examples:
                if ex not in selected_clickbait:
                    selected_clickbait.append(ex)
                    break
        
        # For non-clickbait: pick more formal, news-style examples
        non_clickbait_patterns = [
            lambda x: any(word in x['title'].lower() for word in ['thông báo', 'công bố', 'quyết định']),
            lambda x: any(word in x['title'].lower() for word in ['chính thức', 'khai mạc', 'bế mạc']),
            lambda x: any(word in x['title'].lower() for word in ['hội nghị', 'cuộc họp', 'gặp gỡ']),
            lambda x: any(word in x['title'].lower() for word in ['số liệu', 'báo cáo', 'thống kê']),
            lambda x: len(x['title']) > 60,  # Longer, more formal titles
        ]
        
        for pattern in non_clickbait_patterns:
            matching = [x for x in non_clickbait_examples if pattern(x)]
            if matching and len(selected_non_clickbait) < 5:
                selected_non_clickbait.append(matching[0])
        
        # Fill remaining slots
        while len(selected_non_clickbait) < 5:
            for ex in non_clickbait_examples:
                if ex not in selected_non_clickbait:
                    selected_non_clickbait.append(ex)
                    break
        
        # Combine and arrange strategically (alternating for better learning)
        self.few_shot_examples = []
        for i in range(5):
            if i < len(selected_clickbait):
                self.few_shot_examples.append(selected_clickbait[i])
            if i < len(selected_non_clickbait):
                self.few_shot_examples.append(selected_non_clickbait[i])
        
        logger.info(f"Selected {len(self.few_shot_examples)} strategic few-shot examples:")
        for i, example in enumerate(self.few_shot_examples, 1):
            logger.info(f"  {i}. [{example['label']}] {example['title'][:60]}...")
        
        # Create a separate set for validation (different from training examples)
        self.few_shot_validation_examples = []
        remaining_clickbait = [x for x in clickbait_examples if x not in selected_clickbait]
        remaining_non_clickbait = [x for x in non_clickbait_examples if x not in selected_non_clickbait]
        
        # Add 2 more examples for validation
        if remaining_clickbait:
            self.few_shot_validation_examples.append(remaining_clickbait[0])
        if remaining_non_clickbait:
            self.few_shot_validation_examples.append(remaining_non_clickbait[0])
    
    def setup_models(self, model_name: str, model_path: str):
        """Setup both base and fine-tuned models"""
        logger.info(f"Setting up {model_name} models ({model_path})...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model with quantization
            if 'gemma' in model_name:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map='auto',
                    low_cpu_mem_usage=True
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map='auto',
                    low_cpu_mem_usage=True
                )
            
            self.tokenizers[model_name] = tokenizer
            self.base_models[model_name] = base_model
            
            # Try to load fine-tuned model (LoRA adapter)
            adapter_path = self.adapter_paths[model_name]
            if os.path.exists(adapter_path):
                logger.info(f"Loading LoRA adapter for {model_name} from {adapter_path}")
                finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
                self.finetuned_models[model_name] = finetuned_model
                logger.info(f"✓ Successfully loaded fine-tuned {model_name}")
            else:
                logger.warning(f"LoRA adapter not found for {model_name} at {adapter_path}")
                logger.info(f"Will only evaluate base model for {model_name}")
            
            logger.info(f"✓ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            return False
    
    def create_prompt(self, title: str, few_shot: bool = False) -> str:
        """Create improved prompt for clickbait detection"""
        
        system_prompt = """Bạn là một chuyên gia phân tích tin tức có kinh nghiệm. Nhiệm vụ của bạn là phân loại tiêu đề tin tức thành "clickbait" hoặc "non-clickbait".

ĐỊNH NGHĨA:
- Clickbait: Tiêu đề được thiết kế để khiến người đọc tò mò và phải click vào để biết thông tin, thường dùng từ ngữ cảm xúc mạnh, câu hỏi gây tò mò, hoặc hứa hẹn thông tin mà không tiết lộ cụ thể.
- Non-clickbait: Tiêu đề thông tin rõ ràng, trực tiếp, cung cấp thông tin cụ thể về nội dung mà không cần gây tò mò.

HƯỚNG DẪN: Trả lời chính xác bằng "clickbait" hoặc "non-clickbait"."""

        if few_shot:
            # Enhanced few-shot format with explanations
            examples = "CÁC VÍ DỤ MINH HỌA:\n\n"
            for i, example in enumerate(self.few_shot_examples, 1):
                examples += f"Ví dụ {i}:\n"
                examples += f"Tiêu đề: \"{example['title']}\"\n"
                examples += f"Phân loại: {example['label']}\n\n"
            
            prompt = f"{system_prompt}\n\n{examples}BÂY GIỜ HÃY PHÂN LOẠI:\n\nTiêu đề: \"{title}\"\nPhân loại:"
        else:
            prompt = f"{system_prompt}\n\nTiêu đề: \"{title}\"\nPhân loại:"
            
        return prompt
    
    def generate_prediction(self, model_name: str, prompt: str, use_finetuned: bool = True) -> str:
        """Generate prediction with improved parameters for few-shot"""
        tokenizer = self.tokenizers[model_name]
        
        # Select model
        if use_finetuned and model_name in self.finetuned_models:
            model = self.finetuned_models[model_name]
            model_type = "fine-tuned"
        elif model_name in self.base_models:
            model = self.base_models[model_name]
            model_type = "base"
        else:
            logger.error(f"No model available for {model_name}")
            return "non-clickbait"
        
        # Handle device placement
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
            
        # Improved tokenization for few-shot (longer context)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Improved generation parameters for few-shot
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,  # Increased from 15
                do_sample=True,     # Enable sampling for better diversity
                temperature=0.3,    # Slightly higher temperature for few-shot
                top_p=0.9,         # Add nucleus sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
            )
            
        # Clear cache after each inference
        torch.cuda.empty_cache()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Improved response parsing
        response_lower = response.lower()
        
        # More robust parsing
        if "clickbait" in response_lower and "non-clickbait" not in response_lower:
            return "clickbait"
        elif "non-clickbait" in response_lower or "non clickbait" in response_lower:
            return "non-clickbait"
        elif "không phải clickbait" in response_lower:
            return "non-clickbait"
        elif "là clickbait" in response_lower:
            return "clickbait"
        else:
            # If unclear, try to infer from context
            if any(word in response_lower for word in ['click', 'bấm', 'tò mò', 'gây chú ý']):
                return "clickbait"
            else:
                return "non-clickbait"  # Default to non-clickbait if unclear
    
    def evaluate_model(self, model_name: str, few_shot: bool = False, use_finetuned: bool = True) -> Dict:
        """Evaluate model on test set"""
        model_type = "fine-tuned" if use_finetuned else "base"
        shot_type = "few-shot" if few_shot else "zero-shot"
        
        logger.info(f"Evaluating {model_name} ({shot_type}, {model_type})...")
        
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
            pred = self.generate_prediction(model_name, prompt, use_finetuned=use_finetuned)
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
        
        logger.info(f"Results for {model_name} ({shot_type}, {model_type}):")
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
    
    def run_evaluation_experiments(self):
        """Run evaluation experiments on fine-tuned models"""
        logger.info("Starting fine-tuned models prompting evaluation...")
        
        # Load data
        self.load_data()
        
        # Process each model individually
        logger.info("\n" + "="*70)
        logger.info("FINE-TUNED MODELS PROMPTING EVALUATION")
        logger.info("="*70)
        
        for model_name, model_path in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATING {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Setup models
            success = self.setup_models(model_name, model_path)
            if not success:
                logger.warning(f"✗ Skipping {model_name} due to loading error")
                continue
            
            if model_name not in self.results:
                self.results[model_name] = {}
            
            # Evaluate base model (for comparison)
            logger.info(f"\n>>> Base model evaluation for {model_name}")
            
            # Base Zero-shot
            self.results[model_name]['base_zeroshot'] = self.evaluate_model(
                model_name, few_shot=False, use_finetuned=False
            )
            
            # Base Few-shot
            self.results[model_name]['base_fewshot'] = self.evaluate_model(
                model_name, few_shot=True, use_finetuned=False
            )
            
            # Evaluate fine-tuned model (if available)
            if model_name in self.finetuned_models:
                logger.info(f"\n>>> Fine-tuned model evaluation for {model_name}")
                
                # Fine-tuned Zero-shot
                self.results[model_name]['finetuned_zeroshot'] = self.evaluate_model(
                    model_name, few_shot=False, use_finetuned=True
                )
                
                # Fine-tuned Few-shot
                self.results[model_name]['finetuned_fewshot'] = self.evaluate_model(
                    model_name, few_shot=True, use_finetuned=True
                )
            else:
                logger.warning(f"No fine-tuned model available for {model_name}")
            
            # Memory cleanup
            if model_name in self.finetuned_models:
                del self.finetuned_models[model_name]
            if model_name in self.base_models:
                del self.base_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            torch.cuda.empty_cache()
            logger.info(f"✓ Completed evaluation for {model_name}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"finetuned_prompting_evaluation_{timestamp}.csv"
        self.save_results(self.results, filename)
        
        logger.info(f"\n{'='*70}")
        logger.info("FINE-TUNED MODELS EVALUATION COMPLETED!")
        logger.info(f"{'='*70}")
        
        # Print comparison summary
        self.print_comparison_summary()
    
    def print_comparison_summary(self):
        """Print comparison between base and fine-tuned models"""
        logger.info("\n" + "="*70)
        logger.info("BASE vs FINE-TUNED COMPARISON SUMMARY")
        logger.info("="*70)
        
        for model_name in self.results:
            logger.info(f"\n{model_name.upper()}:")
            
            model_results = self.results[model_name]
            
            # Zero-shot comparison
            if 'base_zeroshot' in model_results and 'finetuned_zeroshot' in model_results:
                base_acc = model_results['base_zeroshot']['accuracy']
                ft_acc = model_results['finetuned_zeroshot']['accuracy']
                improvement = ft_acc - base_acc
                
                base_f1 = model_results['base_zeroshot']['macro_f1']
                ft_f1 = model_results['finetuned_zeroshot']['macro_f1']
                f1_improvement = ft_f1 - base_f1
                
                logger.info(f"  Zero-shot:")
                logger.info(f"    Base accuracy: {base_acc:.4f}")
                logger.info(f"    Fine-tuned accuracy: {ft_acc:.4f}")
                logger.info(f"    Improvement: {improvement:+.4f}")
                logger.info(f"    Base Macro F1: {base_f1:.4f}")
                logger.info(f"    Fine-tuned Macro F1: {ft_f1:.4f}")
                logger.info(f"    F1 Improvement: {f1_improvement:+.4f}")
            
            # Few-shot comparison
            if 'base_fewshot' in model_results and 'finetuned_fewshot' in model_results:
                base_acc = model_results['base_fewshot']['accuracy']
                ft_acc = model_results['finetuned_fewshot']['accuracy']
                improvement = ft_acc - base_acc
                
                base_f1 = model_results['base_fewshot']['macro_f1']
                ft_f1 = model_results['finetuned_fewshot']['macro_f1']
                f1_improvement = ft_f1 - base_f1
                
                logger.info(f"  Few-shot:")
                logger.info(f"    Base accuracy: {base_acc:.4f}")
                logger.info(f"    Fine-tuned accuracy: {ft_acc:.4f}")
                logger.info(f"    Improvement: {improvement:+.4f}")
                logger.info(f"    Base Macro F1: {base_f1:.4f}")
                logger.info(f"    Fine-tuned Macro F1: {ft_f1:.4f}")
                logger.info(f"    F1 Improvement: {f1_improvement:+.4f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run fine-tuned models evaluation
    evaluation = FineTunedPromptingEvaluation()
    evaluation.run_evaluation_experiments() 