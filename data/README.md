# Data Directory

## Dataset: Food101

The Food101 dataset is automatically downloaded via TensorFlow Datasets when you run the training script.

### Dataset Statistics

- **Total Images:** 101,000
- **Training Images:** 75,750 (750 per class)
- **Test Images:** 25,250 (250 per class)
- **Classes:** 101 food categories
- **Image Size:** Variable (resized to 224x224 during preprocessing)
- **Format:** JPEG
- **Total Size:** ~5GB

### Food Categories

The dataset includes 101 food categories such as:
- Apple pie
- Baby back ribs
- Baklava
- Beef carpaccio
- Beignets
- ... (97 more)

Full list available at: https://www.tensorflow.org/datasets/catalog/food101

### Data Structure (after download)
```
~/tensorflow_datasets/
└── food101/
    ├── 3.0.0/
    │   ├── food101-train.tfrecord-*
    │   └── food101-test.tfrecord-*
    └── dataset_info.json
```

### Preprocessing

Images are preprocessed using the following pipeline:
1. Resize to 224x224 pixels
2. Convert from uint8 to float32
3. Normalize (handled by EfficientNet preprocessing)

### Data Augmentation (Training Only)

- Random rotation: ±20°
- Random horizontal flip
- Random zoom: 10%
- Color jittering

### Citation
```bibtex
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```
