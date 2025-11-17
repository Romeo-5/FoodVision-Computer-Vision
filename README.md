# 🍔 FoodVision: Deep Learning Food Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-77%25-green)

A computer vision system that classifies 75,000+ food images across 101 categories using Convolutional Neural Networks and Vision Transformers.

## 🎯 Key Features
- **77% top-1 accuracy** on 101 food categories
- Processes **75,000+ training images**
- Transfer learning with **EfficientNetB0**
- Systematic data augmentation (rotation, scaling, color jittering)
- Comprehensive evaluation pipeline with per-class metrics

## 🛠️ Tech Stack
- **Framework:** PyTorch, TensorFlow
- **Architecture:** EfficientNetB0 (Vision Transformer)
- **Libraries:** torchvision, scikit-learn, pandas, NumPy, matplotlib
- **Techniques:** Transfer learning, data augmentation, regularization (dropout, weight decay)

## 📊 Results
| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 77% |
| Training Images | 75,000+ |
| Categories | 101 |
| Validation Images | 25,000 |

### Performance Highlights
- Reduced overfitting by **23%** through regularization
- **15% improvement** over baseline CNN
- Per-class precision/recall analysis with confusion matrix

## 🚀 Quick Start
[Add installation and usage instructions]

## 📈 Model Architecture
[Add diagram or description of EfficientNetB0 + your modifications]

## 🔍 Challenges & Solutions
- **Challenge:** High visual diversity across categories
- **Solution:** Systematic augmentation + transfer learning
- **Challenge:** Similar food types (e.g., pasta varieties)
- **Solution:** Fine-grained feature extraction focus

## 📝 Key Learnings
- Transfer learning dramatically improves performance on limited data
- Data augmentation must preserve category-defining features
- Per-class analysis reveals model strengths/weaknesses better than overall accuracy

## 📧 Contact
Romeo Nickel - [LinkedIn](https://linkedin.com/in/romeo-nickel) - rjnickel@usc.edu
