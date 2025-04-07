import cv2
import tensorflow as tf
import numpy as np

# Load the trained model (ensure the file path is correct)
model = tf.keras.models.load_model("ml/luviel_model.keras")

# Update labels to match the 4-class output of your model
labels = ["Papules & Pustules", "Black & Whitehead", "Cyst", "Acne"]

def preprocess_image(image):
    """
    Resize and normalize the image.
    Adjust (224, 224) if your model expects a different size.
    """
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def scan_face_from_array(image):
    """
    Process an image (as a NumPy array) and return the formatted prediction result.
    """
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    
    # Construct a result string from the 4 predicted values
    result_text = " | ".join([f"{labels[i]}: {prediction[i]*100:.2f}%" for i in range(len(labels))])
    return result_text

def scan_face(image_file):
    """
    Accepts an image file (file-like object) and returns the prediction result.
    This function is used by the Flask endpoint when an image is uploaded.
    """
    # Read the image file bytes and convert to a NumPy array
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return "Invalid image"
    return scan_face_from_array(image)
