"""
FoodVision Prediction Script
Make predictions on new food images
"""

import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

IMG_SIZE = 224

def load_model(model_path="models/foodvision_final_model"):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img

def predict_image(model, img_path, class_names):
    """Make prediction on a single image"""
    # Load and preprocess
    img = load_and_preprocess_image(img_path)
    img_array = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = tf.argmax(predictions[0])
    confidence = tf.reduce_max(predictions[0])
    
    return predicted_class.numpy(), confidence.numpy()

def plot_prediction(img_path, predicted_class, confidence, class_names):
    """Plot image with prediction"""
    img = plt.imread(img_path)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_names[predicted_class]} "
             f"({confidence*100:.2f}% confidence)", 
             fontsize=16)
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print("Prediction visualization saved to prediction_result.png")
    plt.show()

def main(args):
    """Main prediction pipeline"""
    # Load model
    model = load_model(args.model_path)
    
    # Get class names from Food101
    import tensorflow_datasets as tfds
    ds_info = tfds.builder("food101").info
    class_names = ds_info.features["label"].names
    
    # Make prediction
    predicted_class, confidence = predict_image(
        model, args.image, class_names
    )
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Image: {args.image}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"{'='*50}\n")
    
    # Plot if requested
    if args.plot:
        plot_prediction(args.image, predicted_class, confidence, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict food class from image")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--model-path", type=str, 
                       default="models/foodvision_final_model",
                       help="Path to trained model")
    parser.add_argument("--plot", action="store_true",
                       help="Display prediction visualization")
    
    args = parser.parse_args()
    main(args)
