#!/usr/bin/env python3
"""
Fine-tuning Experiment with QLoRA
================================

Script này thực hiện fine-tuning với QLoRA (Quantized Low-Rank Adaptation):
- 4-bit quantization để giảm memory footprint
- LoRA để fine-tune chỉ một phần nhỏ parameters
- Hỗ trợ cả meta-llama/Llama-3.1-8B-Instruct và google/gemma-7b-it

QLoRA Benefits:
- Memory efficient: ~4x ít memory hơn full fine-tuning
- Fast training: Chỉ update LoRA weights
- Good performance: Comparable results với full fine-tuning
"""

import torch
import json
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLoRAFineTuningExperiment:
    """Experiment class for QLoRA fine-tuning"""
    
    def __init__(self):
        self.models = {
            'meta_llama3': 'meta-llama/Llama-3.1-8B-Instruct',
            'gemma': 'google/gemma-7b-it'
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.tokenizers = {}
        self.base_models = {}
        self.lora_models = {}
        
        self.results = {}
        
        # QLoRA Configuration
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype_fp16=True
        )
        
        # LoRA Configuration
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha scaling
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
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
        """Setup tokenizer and model for QLoRA fine-tuning"""
        logger.info(f"Setting up {model_name} for QLoRA ({model_path})...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load base model with 4-bit quantization
            logger.info(f"Loading {model_name} with 4-bit quantization...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=self.quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Prepare model for k-bit training
            logger.info(f"Preparing {model_name} for k-bit training...")
            base_model = prepare_model_for_kbit_training(base_model)
            
            # Apply LoRA
            logger.info(f"Applying LoRA to {model_name}...")
            peft_model = get_peft_model(base_model, self.lora_config)
            
            # Print trainable parameters
            peft_model.print_trainable_parameters()
            
            self.tokenizers[model_name] = tokenizer
            self.base_models[model_name] = base_model
            self.lora_models[model_name] = peft_model
            
            logger.info(f"✓ Successfully loaded {model_name} with QLoRA")
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
        
        logger.info(f"Created {len(finetuning_data)} training examples")
        return Dataset.from_list(finetuning_data)
    
    def finetune_model(self, model_name: str):
        """Fine-tune a specific model"""
        logger.info(f"\n{'='*60}")
        logger.info(f"FINE-TUNING {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        if model_name not in self.lora_models:
            logger.error(f"Model {model_name} not loaded. Skipping fine-tuning.")
            return False
            
        try:
            peft_model = self.lora_models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Prepare dataset
            dataset = self.prepare_finetuning_data()
            
            def tokenize_function(examples):
                # Handle batched data correctly
                texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
                return tokenizer(
                    texts,
                    truncation=True,
                    padding='max_length',  # Use max_length padding
                    max_length=512,
                    return_tensors=None  # Don't return tensors here, let data collator handle it
                )
            
            logger.info("Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                batch_size=100,  # Process in smaller batches
                remove_columns=['text']  # Remove original text column
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Training arguments optimized for QLoRA
            training_args = TrainingArguments(
                output_dir=f"./qlora_finetuned_{model_name}",
                overwrite_output_dir=True,
                num_train_epochs=3,  # Can increase since QLoRA is memory efficient
                per_device_train_batch_size=4,  # Can increase batch size with QLoRA
                gradient_accumulation_steps=4,  # Reduced since batch size is higher
                warmup_steps=100,
                logging_steps=50,
                save_steps=500,
                eval_strategy="no",
                learning_rate=2e-4,  # Higher learning rate for LoRA
                fp16=True,
                dataloader_num_workers=0,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb
                gradient_checkpointing=True,
                dataloader_pin_memory=False,
                ddp_find_unused_parameters=False,
                optim="paged_adamw_8bit",  # Use 8-bit optimizer for QLoRA
                max_grad_norm=0.3,  # Gradient clipping
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
            )
            
            # Trainer
            trainer = Trainer(
                model=peft_model,  # Use PEFT model for training
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Train
            logger.info(f"Starting QLoRA training for {model_name}...")
            start_time = time.time()
            trainer.train()
            train_time = time.time() - start_time
            
            logger.info(f"QLoRA training completed in {train_time/60:.1f} minutes")
            
            # Save LoRA adapter
            logger.info(f"Saving LoRA adapter for {model_name}...")
            peft_model.save_pretrained(f"./qlora_adapter_{model_name}")
            tokenizer.save_pretrained(f"./qlora_adapter_{model_name}")
            
            # Store fine-tuned model
            self.lora_models[model_name] = peft_model  # Update the model in lora_models
            
            logger.info(f"✓ Successfully fine-tuned {model_name} with QLoRA")
            return True
            
        except Exception as e:
            logger.error(f"Error fine-tuning {model_name}: {str(e)}")
            return False
    
    def generate_prediction(self, model_name: str, prompt: str, use_finetuned: bool = True) -> str:
        """Generate prediction using the model (base or LoRA fine-tuned)"""
        tokenizer = self.tokenizers[model_name]
        
        # Use LoRA model if available and requested
        if use_finetuned and model_name in self.lora_models:
            model = self.lora_models[model_name]
        elif model_name in self.base_models:
            model = self.base_models[model_name]
        else:
            logger.error(f"No model available for {model_name}")
            return "non-clickbait"
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Handle device placement for quantized models
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
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
    
    def evaluate_model(self, model_name: str, few_shot: bool = False, use_finetuned: bool = True) -> Dict:
        """Evaluate model on test set"""
        model_type = "fine-tuned" if use_finetuned else "pre-trained"
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
    
    def run_finetuning_experiments(self):
        """Run fine-tuning experiments - one model at a time to save VRAM"""
        logger.info("Starting fine-tuning experiments...")
        
        # Disable wandb
        os.environ["WANDB_DISABLED"] = "true"
        
        # Load data
        self.load_data()
        
        # Process each model individually to optimize VRAM
        logger.info("\n" + "="*70)
        logger.info("FINE-TUNING EXPERIMENTS")
        logger.info("="*70)
        
        for model_name, model_path in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Clear all memory before loading new model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Setup model
            success = self.setup_model(model_name, model_path)
            if not success:
                logger.warning(f"✗ Skipping {model_name} due to loading error")
                continue
            
            logger.info(f"✓ {model_name} loaded successfully")
            
            # Fine-tune the model
            logger.info(f"\n>>> Fine-tuning {model_name}")
            finetune_success = self.finetune_model(model_name)
            
            if finetune_success:
                logger.info(f"✓ {model_name} fine-tuned successfully")
                
                # Initialize results for this model
                if model_name not in self.results:
                    self.results[model_name] = {}
                
                # Evaluate fine-tuned model
                logger.info(f"\n>>> Evaluating fine-tuned {model_name}")
                
                # Zero-shot with fine-tuned model
                logger.info(f"Zero-shot evaluation for fine-tuned {model_name}")
                self.results[model_name]['finetuned_zeroshot'] = self.evaluate_model(
                    model_name, few_shot=False, use_finetuned=True
                )
                
                # Few-shot with fine-tuned model
                logger.info(f"Few-shot evaluation for fine-tuned {model_name}")
                self.results[model_name]['finetuned_fewshot'] = self.evaluate_model(
                    model_name, few_shot=True, use_finetuned=True
                )
                
                logger.info(f"✓ Completed evaluation for fine-tuned {model_name}")
            else:
                logger.warning(f"✗ {model_name} fine-tuning failed")
            
            # Aggressive memory cleanup
            if model_name in self.lora_models:
                del self.lora_models[model_name]
            if model_name in self.base_models:
                del self.base_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"✓ Aggressive memory cleanup completed after {model_name}")
        
        # Save results
        if self.results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finetuned_results_{timestamp}.csv"
            self.save_results(self.results, filename)
            
            logger.info(f"\n{'='*70}")
            logger.info("FINE-TUNING EXPERIMENTS COMPLETED!")
            logger.info(f"{'='*70}")
            
            # Print summary
            self.print_summary()
        else:
            logger.warning("No results to save - all experiments failed")
    
    def print_summary(self):
        """Print summary of all results"""
        logger.info("\n" + "="*70)
        logger.info("FINE-TUNING EXPERIMENT SUMMARY")
        logger.info("="*70)
        
        for model_name, model_results in self.results.items():
            logger.info(f"\n{model_name.upper()} (FINE-TUNED):")
            for experiment_type, metrics in model_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    logger.info(f"  {experiment_type:25}: "
                               f"Acc={metrics['accuracy']:.4f}, "
                               f"Macro F1={metrics['macro_f1']:.4f}, "
                               f"Weighted F1={metrics['weighted_f1']:.4f}")
        
        logger.info(f"\n{'='*70}")
        logger.info("ALL EXPERIMENTS COMPLETED!")
        logger.info("="*70)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run fine-tuning experiments
    experiment = QLoRAFineTuningExperiment()
    experiment.run_finetuning_experiments() 