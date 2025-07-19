# Clickbait Detection: Prompting vs Fine-tuning

Dự án nghiên cứu so sánh hiệu quả của **Prompting** và **Fine-tuning** trong bài toán phát hiện clickbait tiếng Việt sử dụng các mô hình ngôn ngữ lớn (LLMs).

## 🎯 Kết quả chính

| Phương pháp | Mô hình tốt nhất | Accuracy | Macro F1 | Ưu điểm |
|-------------|------------------|----------|----------|---------|
| **Prompting** | Gemma 7B (Few-shot) | **75.6%** | **74.0%** | Không cần training, nhanh |
| **Fine-tuning** | Llama 3.1 8B (Zero-shot) | **85.7%** | **83.2%** | Hiệu suất cao nhất |

> **Kết luận**: Fine-tuning với QLoRA vượt trội hơn prompting ~10% accuracy, đáng giá với chi phí training hợp lý.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Thí nghiệm](#thí-nghiệm)
- [Kết quả](#kết-quả)
- [Đóng góp](#đóng-góp)

## 🎯 Tổng quan

### Mục tiêu
- So sánh hiệu quả của **Zero-shot/Few-shot Prompting** vs **Fine-tuning với QLoRA**
- Đánh giá trên dataset clickbait tiếng Việt
- Phân tích trade-off giữa hiệu suất và chi phí tính toán

### Mô hình được sử dụng
- **Meta Llama 3.1 8B Instruct**
- **Google Gemma 7B IT**

### Phương pháp
1. **Prompting Approaches:**
   - Zero-shot prompting
   - Few-shot prompting (3-6 examples)
   - Improved prompting với context tốt hơn

2. **Fine-tuning Approaches:**
   - QLoRA (Quantized Low-Rank Adaptation)
   - 4-bit quantization để tiết kiệm memory
   - LoRA rank 16, alpha 32

## 📁 Cấu trúc dự án

```
prompting-finetune/
├── README.md                    # File này
├── requirements.txt             # Dependencies
├── 
├── src/                        # Source code chính
│   ├── __init__.py
│   ├── experiments/            # Các thí nghiệm
│   │   ├── __init__.py
│   │   ├── clickbait_experiment.py      # Thí nghiệm chính
│   │   ├── finetuning_experiment.py     # Fine-tuning với QLoRA
│   │   ├── pretrained_prompting.py      # Zero/Few-shot prompting
│   │   ├── pretrained_prompting_improved.py  # Prompting cải tiến
│   │   └── evaluate_finetuned_prompting.py   # Đánh giá mô hình
│   ├── models/                 # Model utilities
│   │   └── __init__.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── data_loader.py      # Data loading utilities
│       └── metrics.py          # Evaluation metrics
│
├── data/                       # Dữ liệu
│   ├── simple_dataset/         # Dataset clickbait
│   │   ├── train/              # Training data
│   │   ├── val/                # Validation data
│   │   └── test/               # Test data
│   └── processed/              # Processed data
│
├── scripts/                    # Scripts chạy thí nghiệm
│   ├── run_experiment.py       # Script test đơn giản
│   ├── run_improved_evaluation.py  # Evaluation cải tiến
│   └── run_quick_test.py       # Quick test
│
├── results/                    # Kết quả thí nghiệm
│   ├── experiments/            # CSV results
│   ├── models/                 # Saved models
│   └── logs/                   # Log files
│
├── docs/                       # Documentation
└── tests/                      # Unit tests
```

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA-capable GPU (khuyến nghị 16GB+ VRAM)
- 32GB+ RAM

### Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd prompting-finetune

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### Cấu hình Hugging Face
```bash
# Đăng nhập Hugging Face để truy cập các mô hình
huggingface-cli login
```

## 💻 Sử dụng

### 1. Test nhanh hệ thống
```bash
python scripts/run_experiment.py
```

### 2. Chạy thí nghiệm Prompting
```bash
# Zero-shot và Few-shot prompting
python src/experiments/pretrained_prompting.py

# Prompting cải tiến
python src/experiments/pretrained_prompting_improved.py
```

### 3. Chạy thí nghiệm Fine-tuning
```bash
# Fine-tuning với QLoRA
python src/experiments/finetuning_experiment.py
```

### 4. So sánh kết quả
```bash
# Đánh giá và so sánh các phương pháp
python src/experiments/evaluate_finetuned_prompting.py
```

## 🧪 Thí nghiệm

### Dataset
- **Nguồn**: Dataset clickbait tiếng Việt
- **Tổng kích thước**: 3,414 samples
- **Phân chia**:
  - **Train**: 2,389 samples (745 clickbait - 31.2%, 1,644 non-clickbait - 68.8%)
  - **Validation**: 513 samples (160 clickbait - 31.2%, 353 non-clickbait - 68.8%)
  - **Test**: 512 samples (160 clickbait - 31.2%, 352 non-clickbait - 68.8%)
- **Labels**: `clickbait` và `non-clickbait`
- **Đặc điểm**: Dataset cân bằng với tỷ lệ clickbait/non-clickbait khoảng 1:2

### Metrics đánh giá
- **Accuracy**: Độ chính xác tổng thể
- **Precision/Recall/F1**: Cho từng class
- **Macro F1**: Trung bình F1 của các class

### Cấu hình thí nghiệm

#### Prompting
- **Temperature**: 0.1 (deterministic)
- **Max tokens**: 10
- **Few-shot examples**: 3-6 examples

#### Fine-tuning  
- **Method**: QLoRA
- **Quantization**: 4-bit
- **LoRA rank**: 16
- **LoRA alpha**: 32
- **Learning rate**: 2e-4
- **Batch size**: 4
- **Epochs**: 3

## 📊 Kết quả

Kết quả chi tiết được lưu trong thư mục `results/experiments/`:
- `pretrained_prompting_results_20250712_112415.csv` - Kết quả prompting experiments
- `finetuned_results_20250712_154101.csv` - Kết quả fine-tuning experiments

### Prompting Results (`pretrained_prompting_results_20250712_112415.csv`)

| Model | Experiment | Accuracy | Macro F1 | Clickbait F1 | Non-Clickbait F1 |
|-------|------------|----------|----------|--------------|------------------|
| **Llama 3.1 8B** | Zero-shot | 0.746 | 0.644 | 0.454 | 0.835 |
| **Llama 3.1 8B** | Few-shot | 0.754 | 0.740 | 0.680 | 0.800 |
| **Gemma 7B** | Zero-shot | 0.385 | 0.363 | 0.481 | 0.245 |
| **Gemma 7B** | Few-shot | 0.756 | 0.693 | 0.555 | 0.832 |

### Fine-tuning Results (`finetuned_results_20250712_154101.csv`)

| Model | Experiment | Accuracy | Macro F1 | Clickbait F1 | Non-Clickbait F1 |
|-------|------------|----------|----------|--------------|------------------|
| **Llama 3.1 8B** | Zero-shot | 0.857 | 0.832 | 0.767 | 0.897 |
| **Llama 3.1 8B** | Few-shot | 0.832 | 0.817 | 0.765 | 0.869 |
| **Gemma 7B** | Zero-shot | 0.834 | 0.812 | 0.748 | 0.876 |
| **Gemma 7B** | Few-shot | 0.844 | 0.823 | 0.763 | 0.883 |

### Phân tích kết quả

#### 🏆 **Fine-tuning vượt trội hơn Prompting**
- **Accuracy**: Fine-tuning đạt 83-86% vs Prompting 38-76%
- **Macro F1**: Fine-tuning đạt 81-83% vs Prompting 36-74%
- **Cải thiện đáng kể** trong việc phát hiện clickbait

#### 🔍 **So sánh chi tiết**
| Phương pháp | Best Accuracy | Best Macro F1 | Training Time | Memory Usage |
|-------------|---------------|---------------|---------------|--------------|
| **Prompting Zero-shot** | 0.746 (Llama) | 0.644 (Llama) | 0 | Low |
| **Prompting Few-shot** | 0.756 (Gemma) | 0.740 (Llama) | 0 | Low |
| **Fine-tuning Zero-shot** | 0.857 (Llama) | 0.832 (Llama) | ~2 hours | Medium |
| **Fine-tuning Few-shot** | 0.844 (Gemma) | 0.823 (Gemma) | ~2 hours | Medium |

#### 📈 **Insights**
- **Llama 3.1 8B** tổng thể tốt hơn **Gemma 7B** trong prompting
- **Fine-tuning** giúp cả hai mô hình đạt hiệu suất tương đương cao
- **Few-shot prompting** cải thiện đáng kể so với zero-shot (đặc biệt với Gemma)
- **QLoRA fine-tuning** mang lại kết quả ấn tượng với chi phí tính toán hợp lý

## 🏆 Ranking tất cả các phương pháp

| Rank | Phương pháp | Mô hình | Accuracy | Macro F1 | Clickbait F1 | Non-Clickbait F1 |
|------|-------------|---------|----------|----------|--------------|------------------|
| 1 | Fine-tuning Zero-shot | Llama 3.1 8B | **85.7%** | **83.2%** | 76.7% | 89.7% |
| 2 | Fine-tuning Few-shot | Gemma 7B | 84.4% | 82.3% | 76.3% | 88.3% |
| 3 | Fine-tuning Zero-shot | Gemma 7B | 83.4% | 81.2% | 74.8% | 87.6% |
| 4 | Fine-tuning Few-shot | Llama 3.1 8B | 83.2% | 81.7% | 76.5% | 86.9% |
| 5 | Prompting Few-shot | Gemma 7B | 75.6% | 69.3% | 55.5% | 83.2% |
| 6 | Prompting Few-shot | Llama 3.1 8B | 75.4% | 74.0% | 68.0% | 80.0% |
| 7 | Prompting Zero-shot | Llama 3.1 8B | 74.6% | 64.4% | 45.4% | 83.5% |
| 8 | Prompting Zero-shot | Gemma 7B | 38.5% | 36.3% | 48.1% | 24.5% |

## 📊 Phân tích chi tiết

### Fine-tuning vs Prompting
- **Fine-tuning** vượt trội với accuracy 83-86% vs prompting 38-76%
- **Cải thiện lớn nhất**: Gemma 7B từ 38.5% (zero-shot prompting) lên 84.4% (few-shot fine-tuning)
- **Llama 3.1 8B** ổn định hơn trong prompting, nhưng cả hai mô hình đều đạt hiệu suất tương đương sau fine-tuning

### Zero-shot vs Few-shot
#### Prompting:
- **Few-shot** cải thiện đáng kể so với zero-shot
- Gemma 7B: 38.5% → 75.6% (+37.1%)
- Llama 3.1 8B: 74.6% → 75.4% (+0.8%)

#### Fine-tuning:
- **Zero-shot** thường tốt hơn few-shot một chút
- Có thể do overfitting với few-shot examples

### Phát hiện Clickbait vs Non-Clickbait
- **Non-clickbait F1** luôn cao hơn **Clickbait F1** (do dataset imbalanced)
- Fine-tuning cải thiện đáng kể khả năng phát hiện clickbait
- Prompting gặp khó khăn với class clickbait (minority class)

## 💡 Insights và Khuyến nghị

### Khi nào dùng Prompting:
- ✅ Cần kết quả nhanh, không có thời gian training
- ✅ Dataset nhỏ hoặc không có GPU mạnh
- ✅ Prototype hoặc proof-of-concept
- ⚠️ Chấp nhận accuracy thấp hơn ~10%

### Khi nào dùng Fine-tuning:
- ✅ Cần hiệu suất cao nhất
- ✅ Có GPU và thời gian training (~2 hours)
- ✅ Production system
- ✅ Dataset đủ lớn (>1000 samples)

### Lựa chọn mô hình:
- **Llama 3.1 8B**: Tốt hơn cho prompting, đặc biệt zero-shot
- **Gemma 7B**: Cải thiện mạnh với fine-tuning, hiệu suất tương đương Llama sau fine-tuning
- **QLoRA**: Hiệu quả cho fine-tuning với memory hạn chế

## 🔧 Cấu hình tốt nhất

### Prompting:
```python
# Few-shot với 3-6 examples
temperature = 0.1
max_new_tokens = 10
model = "meta-llama/Llama-3.1-8B-Instruct"  # cho zero-shot
model = "google/gemma-7b-it"  # cho few-shot
```

### Fine-tuning:
```python
# QLoRA configuration
lora_r = 16
lora_alpha = 32
learning_rate = 2e-4
batch_size = 4
epochs = 3
quantization = "4bit"
```

## 📅 Thời gian thí nghiệm
- **Prompting experiments**: 12/07/2025 11:24
- **Fine-tuning experiments**: 12/07/2025 15:41
- **Total runtime**: ~4 hours (bao gồm cả training time)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- **Author**: Nguyen Minh Y   
- **Email**: nguyenminhy7714@gmail.com
- **Project Link**: [https://github.com/blanatole/ViClickbait-2025-Finetune-LLMs](https://github.com/blanatole/ViClickbait-2025-Finetune-LLMs)

## 🙏 Acknowledgments

- Hugging Face Transformers
- Meta Llama 3.1
- Google Gemma
- QLoRA paper authors
