import os
import numpy as np
import cv2
import keras

# Path to the directory containing the models
models_directory = "content/model/medium_project"

# List all files in the directory with .keras extension
model_files = [file for file in os.listdir(models_directory) if file.endswith('.keras')]

# Load all models into a list
models = [keras.models.load_model(os.path.join(models_directory, model_file),compile=False) for model_file in model_files]

# Convert the list of models to a NumPy array (optional)
models_array = np.array(models)

# Define label dictionary
label_dict = {
    'Ain': 'ع', 'alif': 'ا', 'ba': 'ب', 'daad': 'ض', 'dal': 'د', 'fa': 'ف',
    'gem': 'ج', 'gen': 'غ', 'ha': 'هـ', 'haa': 'ح', 'kaf': 'ك', 'khaa': 'خ',
    'lam': 'ل', 'lam alif': 'لا', 'mim': 'م', 'nun': 'ن', 'qaf': 'ق', 'raa': 'ر',
    'saad': 'ص', 'shen': 'ش', 'sin': 'س', 'taa': 'ت', 'tah': 'ط', 'thaa': 'ث',
    'waw': 'و', 'yaa': 'ي', 'zah': 'ظ', 'zal': 'ذ', 'zin': 'ز'
}

# Function to preprocess image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to match the input shape of the model (assuming 32x32)
    resized_img = cv2.resize(gray, (32, 32))
    
    # Normalize image
    normalized_img = resized_img / 255.0
    
    # Convert to numpy array and reshape
    img_array = keras.preprocessing.image.img_to_array(normalized_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict image label
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    
    # Aggregate predictions from all models
    predictions = [model.predict(img_array) for model in models_array]
    
    # Average predictions from all models
    avg_prediction = np.mean(predictions, axis=0)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(avg_prediction)
    
    # Map the index to label
    predicted_label = list(label_dict.keys())[predicted_index]
    
    return predicted_label, label_dict[predicted_label]

# Example usage
image_path = 'cropped_169_36.png'  # Replace with your image path
predicted_label, predicted_char = predict_image(image_path)
print(f"Predicted Label: {predicted_label}, Character: {predicted_char}")
