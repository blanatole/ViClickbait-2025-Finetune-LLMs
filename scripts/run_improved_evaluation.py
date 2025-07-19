#!/usr/bin/env python3
"""
Run Improved Fine-tuned Models Evaluation
=======================================

Script to run the improved evaluation with better few-shot prompting
"""

import sys
import os
from evaluate_finetuned_prompting import FineTunedPromptingEvaluation

def main():
    print("="*80)
    print("RUNNING IMPROVED FINE-TUNED MODELS EVALUATION")
    print("="*80)
    print()
    print("🔧 Improvements made:")
    print("  ✓ Increased few-shot examples: 3+3 → 5+5")
    print("  ✓ Strategic example selection (diverse patterns)")
    print("  ✓ Enhanced prompt format with clear structure")
    print("  ✓ Better generation parameters (max_new_tokens: 15→25)")
    print("  ✓ Improved response parsing")
    print("  ✓ Longer context window (max_length: 512→1024)")
    print()
    print("Expected improvement: Few-shot should now outperform zero-shot!")
    print()
    
    # Initialize evaluator
    evaluator = FineTunedPromptingEvaluation()
    
    # Run evaluation
    try:
        evaluator.run_evaluation_experiments()
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print("📊 Check the results CSV file for detailed metrics")
        print("📈 Few-shot should now show improved performance over zero-shot")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("Please check the logs for more details")
        sys.exit(1)

if __name__ == "__main__":
    main() 