import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np
import tensorflow as tf



# Path to the directory containing the models
models_directory = "Model"

# List all files in the directory with .keras extension
model_files = [file for file in os.listdir(models_directory) if file.endswith('.keras')]

# Load all models into a list
models = [keras.models.load_model(os.path.join(models_directory, model_file),compile=False) for model_file in model_files]

# Convert the list of models to a NumPy array (optional)
models_array = np.array(models)

# Display the loaded models
for idx, model in enumerate(models_array):
    print(f"\nModel {idx+1} ({model_files[idx]}):")
    
    # Menampilkan ringkasan model
    print("Model Summary:")
    model.summary()
    
    # Menampilkan informasi tentang arsitektur model
    print("\nModel Architecture:")
    model_config = model.get_config()
    print(model_config)
    
    # Menampilkan informasi tentang bobot model
    print("\nModel Weights:")
    weights = model.get_weights()
    print(f"Number of weight tensors: {len(weights)}")
    for i, weight in enumerate(weights):
        print(f"Weight {i+1} shape: {weight.shape}")
    
    # Menampilkan optimizer yang digunakan (jika tersedia)
    # if model.optimizer:
    #     print("\nOptimizer Information:")
    #     print(f"Optimizer: {model.optimizer.get_config()}")
    # else:
    #     print("\nNo optimizer information available.")
    
    # # Menampilkan metrik dan loss yang digunakan (jika tersedia)
    # if model.compiled_loss:
    #     print("\nLoss Information:")
    #     print(f"Losses: {model.compiled_loss}")

    # if model.compiled_metrics:
    #     print("\nMetrics Information:")
    #     print(f"Metrics: {model.compiled_metrics}")