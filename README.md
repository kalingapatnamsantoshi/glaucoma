# 🩺 Kidney Stone Detector using Machine Learning
# 📌 Overview

Kidney stones are a common health problem and early detection is very important.
This project uses Machine Learning / Deep Learning to detect whether a kidney image contains a stone or is normal.

# The system takes kidney images as input and predicts:
Kidney Stone
Normal Kidney

This project is built mainly for learning, practice, and academic purposes.

# 🎯 Features

Detects kidney stones from images
Simple and beginner-friendly implementation
Uses deep learning (CNN)
Easy to run and test

# 🛠 Technologies Used

Python
TensorFlow / Keras
NumPy
OpenCV
Matplotlib
Scikit-learn

📁 Project Structure
kidney_Stone_Detection/
│
├── Dataset/
│   ├── Stone/
│   └── Normal/
|__Model/
| |__(.h5 file will addedd after training the model
| 
│
├── app.py
├── train.py
├── test_model.py
├── kidney_stone_detector.h5
├── requirements.txt
├── README.md

# 📊 Dataset

Contains kidney medical images

Two classes:
Stone – images with kidney stones
Normal – images without kidney stones
Dataset is not included in this repository for learning and testing

⚠️ Dataset is used only for educational purposes.

# ⚙️ How the Project Works

Load kidney images from the dataset
Preprocess images (resize, normalize)
Train a CNN model
Save the trained model (.h5)
Test the model on new images

# 🚀 How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/your-username/kidney_Stone_Detection.git
cd kidney_Stone_Detection

Step 2: Install Required Libraries
pip install -r requirements.txt

Step 3: Train the Model
python train.py

Step 4: Test the Model
python test_model.py

Step 5: Run the model:
Streamlit run app.py

📈 Model Information

Model Type: Convolutional Neural Network (CNN)
Loss Function: Binary Crossentropy

