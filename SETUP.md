# Setup Instructions

## Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 10GB+ free disk space

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Romeo-5/FoodVision.git
cd FoodVision
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Dataset

The Food101 dataset is automatically downloaded via TensorFlow Datasets when you run the script.

**Dataset Details:**
- 101 food categories
- 75,750 training images (750 per class)
- 25,250 test images (250 per class)
- Total size: ~5GB

## Running the Model

### Training from Scratch
```bash
python foodvision_train.py
```

### Using Pre-trained Model
Download the pre-trained EfficientNetB0 model:
```bash
wget https://storage.googleapis.com/ztm_tf_course/food_vision/07_efficientnetb0_feature_extract_model_mixed_precision.zip
unzip 07_efficientnetb0_feature_extract_model_mixed_precision.zip -d models/
```

### Inference
```bash
python predict.py --image path/to/image.jpg
```

## Training Configuration

- **Base Model:** EfficientNetB0
- **Input Size:** 224x224x3
- **Batch Size:** 32
- **Initial Learning Rate:** 0.001 (feature extraction), 0.0001 (fine-tuning)
- **Mixed Precision:** Enabled (requires GPU with compute capability 7.0+)

## Expected Results

| Phase | Epochs | Accuracy | Time (V100) |
|-------|--------|----------|-------------|
| Feature Extraction | 3 | ~70% | ~20 min |
| Fine-Tuning | 5-10 | ~77% | ~2 hours |

## Troubleshooting

**Issue:** `CUDA out of memory`
- **Solution:** Reduce batch size in the training script

**Issue:** Mixed precision not working
- **Solution:** Check GPU compatibility (`nvidia-smi -L`)

**Issue:** Dataset download fails
- **Solution:** Ensure stable internet connection and sufficient disk space

## GPU Requirements

Recommended: NVIDIA V100, T4, or better
Minimum: NVIDIA GPU with CUDA compute capability 7.0+
