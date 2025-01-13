# Face Recognition with Flask

This project demonstrates a Flask-based web application for face recognition using the LBPH (Local Binary Patterns Histogram) algorithm. It detects and recognizes faces in uploaded images and annotates them with a label (e.g., "Rajinikanth" or "Unknown").

---

## Features
- Upload an image to detect and recognize faces.
- Annotate recognized faces in the image with bounding boxes and labels.
- Save and display the processed image on a result page.
- Train the recognizer with images of Rajinikanth.

---

## Prerequisites
Make sure you have Python 3.7 or higher installed on your system.

---

## Required Libraries
Install the required Python libraries using the following command:
```bash
pip install flask opencv-python opencv-contrib-python numpy
```

---

## Project Directory Structure
```
project/
├── app.py                  # Main application script
├── templates/              # HTML templates for the web app
│   ├── index.html          # Home page template
│   └── result.html         # Result page template
├── static/                 # Static files served by Flask
│   ├── index_style/        # Styles for index.html
│   └── processed/          # Folder for processed images
├── uploads/                # Folder for uploaded images
├── training_images/        # Folder for Rajinikanth's training images
├── models/                 # Folder for storing the trained model
└── README.md               # Project documentation
```

---

## Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Directory Structure**
   Ensure the following folders exist:
   - `uploads/`
   - `training_images/`
   - `static/processed/`
   - `models/`

   These folders are automatically created if they don't exist when the application runs.

4. **Train the Recognizer**
   Place training images of Rajinikanth in the `training_images/` folder and run the application. The model will train automatically if the `rajinikanth_recognizer.yml` file is missing in the `models/` folder.

5. **Run the Application**
   Start the Flask development server:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Usage

1. **Upload an Image**
   - Navigate to the homepage.
   - Upload an image file.
   - Click "Submit."

2. **View Results**
   - The app processes the image and redirects you to the result page.
   - The processed image will display annotated faces with labels.

3. **Back to Home**
   - Click the "Back to Home" link to return to the homepage.

---

## Key Functions

1. **`process_image(filepath, filename)`**
   - Detects faces in the uploaded image using Haar Cascade.
   - Recognizes faces using the LBPH recognizer.
   - Annotates the faces and saves the processed image.

2. **`train_rajinikanth_recognizer()`**
   - Trains the LBPH recognizer with Rajinikanth's images.
   - Saves the trained model as `rajinikanth_recognizer.yml` in the `models/` folder.

---

## API Endpoints

1. **`/`**
   - Method: `GET`
   - Renders the homepage (`index.html`).

2. **`/upload`**
   - Method: `POST`
   - Handles image uploads and processes them.

3. **`/result/<filename>`**
   - Method: `GET`
   - Displays the processed image on the result page.

---

## Notes
- The recognizer will label faces as "Rajinikanth" if confidence is below the threshold (80); otherwise, it will label them as "Unknown."
- Training images should be clear, frontal face images for best results.

---

## Troubleshooting

1. **Image Not Found**
   - Ensure the uploaded file is a valid image.
   - Check that the image is saved correctly in the `uploads/` folder.

2. **Model Not Found**
   - Place training images in the `training_images/` folder and restart the application to train the recognizer.

3. **Static Files Not Loading**
   - Ensure the processed images are saved in the `static/processed/` folder.

---

## Future Enhancements
- Add support for multiple labels and training data for other individuals.
- Improve UI design with Bootstrap or similar frameworks.
- Implement face alignment for better recognition accuracy.

---

## License
This project is open-source and available under the MIT License.

