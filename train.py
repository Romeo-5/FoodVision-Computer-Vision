"""
FoodVision Training Script
Trains EfficientNetB0 on Food101 dataset using transfer learning
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, mixed_precision
import argparse
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 3
FINE_TUNE_EPOCHS = 100
BASE_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001

def setup_mixed_precision():
    """Enable mixed precision training for faster performance"""
    mixed_precision.set_global_policy("mixed_float16")
    print(f"Mixed precision enabled: {mixed_precision.global_policy()}")

def load_data():
    """Load and return Food101 dataset"""
    print("Loading Food101 dataset...")
    (train_data, test_data), ds_info = tfds.load(
        name="food101",
        split=["train", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    class_names = ds_info.features["label"].names
    print(f"Loaded {len(class_names)} classes")
    
    return train_data, test_data, class_names

def preprocess_img(image, label, img_shape=IMG_SIZE):
    """Preprocess images for model input"""
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label

def prepare_dataset(dataset, batch_size=BATCH_SIZE, shuffle=True):
    """Prepare dataset with preprocessing, batching, and prefetching"""
    dataset = dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def create_model(num_classes, trainable=False):
    """Create EfficientNetB0 model with custom head"""
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    # Base model
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = trainable
    
    # Functional model
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype=tf.float32, 
                                name="softmax_float32")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def create_callbacks(experiment_name):
    """Create training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"training_logs/{experiment_name}_{timestamp}"
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            f"checkpoints/{experiment_name}_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    return callbacks

def main(args):
    """Main training pipeline"""
    # Setup
    setup_mixed_precision()
    
    # Load data
    train_data, test_data, class_names = load_data()
    train_data = prepare_dataset(train_data)
    test_data = prepare_dataset(test_data, shuffle=False)
    
    # Create model
    print("\n--- Phase 1: Feature Extraction ---")
    model = create_model(num_classes=len(class_names), trainable=False)
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(BASE_LEARNING_RATE),
        metrics=["accuracy"]
    )
    
    # Feature extraction training
    history_fe = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS,
        validation_data=test_data,
        validation_steps=int(0.15 * len(test_data)),
        callbacks=create_callbacks("feature_extraction")
    )
    
    # Evaluate
    results_fe = model.evaluate(test_data)
    print(f"\nFeature Extraction Results - Loss: {results_fe[0]:.4f}, "
          f"Accuracy: {results_fe[1]:.4f}")
    
    if args.fine_tune:
        print("\n--- Phase 2: Fine-Tuning ---")
        # Unfreeze base model
        model.layers[1].trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
            metrics=["accuracy"]
        )
        
        # Fine-tune
        history_ft = model.fit(
            train_data,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=test_data,
            validation_steps=int(0.15 * len(test_data)),
            callbacks=create_callbacks("fine_tuning")
        )
        
        # Final evaluation
        results_ft = model.evaluate(test_data)
        print(f"\nFine-Tuning Results - Loss: {results_ft[0]:.4f}, "
              f"Accuracy: {results_ft[1]:.4f}")
    
    # Save final model
    model.save("models/foodvision_final_model")
    print("\nModel saved to models/foodvision_final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FoodVision model")
    parser.add_argument("--fine-tune", action="store_true", 
                       help="Perform fine-tuning after feature extraction")
    args = parser.parse_args()
    
    main(args)
