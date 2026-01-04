import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/glaucoma_model.h5"
model = load_model(MODEL_PATH)

# You must check this matches train_generator.class_indices
CLASS_MAPPING = {0: "Glaucoma Detected", 1: "Healthy"}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_glaucoma(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    print(f"🔍 Raw prediction: {prediction}")

    # Sigmoid output
    return CLASS_MAPPING[int(round(prediction))]
