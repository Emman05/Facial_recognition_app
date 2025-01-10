import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Setup paths
UPLOAD_FOLDER = 'uploads'
TRAINING_IMAGES_FOLDER = 'training_images'  # Folder containing images of Rajinikanth
PROCESSED_FOLDER = 'static/processed'
MODEL_FOLDER = 'models'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_IMAGES_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained model for Rajinikanth
model_path = os.path.join(MODEL_FOLDER, 'rajinikanth_recognizer.yml')
if os.path.exists(model_path):
    recognizer.read(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image file uploaded!", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file!", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the uploaded image and save the result
    result_path = process_image(filepath, file.filename)

    if result_path:
        return redirect(url_for('show_result', filename=result_path))
    else:
        return "Error processing image", 500

@app.route('/result/<filename>')
def show_result(filename):
    # Render the result.html page with the processed image filename
    return render_template('result.html', filename=filename)

def process_image(filepath, filename):
    image = cv2.imread(filepath)
    if image is None:
        print("Error: Image not found or unable to load!")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected!")
        return filename

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Recognize the face using the trained model
        label, confidence = recognizer.predict(face_roi)
        if confidence < 80:  # Threshold for recognizing Rajinikanth
            name = "Rajinikanth"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw a rectangle and label the face
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    result_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(result_path, image)
    print(f"Processed image saved at {result_path}")
    return filename

def train_rajinikanth_recognizer():
    face_samples = []
    ids = []

    # Load training images of Rajinikanth
    for image_name in os.listdir(TRAINING_IMAGES_FOLDER):
        image_path = os.path.join(TRAINING_IMAGES_FOLDER, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_samples.append(face_roi)
            ids.append(0)  # Single label for Rajinikanth

    # Train the recognizer with Rajinikanth's face samples
    recognizer.train(face_samples, np.array(ids))
    recognizer.save(model_path)
    print("Model trained successfully for Rajinikanth.")

if __name__ == '__main__':
    # Train the model if it doesn't exist
    if not os.path.exists(model_path):
        train_rajinikanth_recognizer()

    app.run(debug=True)
