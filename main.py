import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Define class labels
class_labels = {
    0: 'Baked Potato',
    1: 'Burger',
    2: 'Crispy Chicken',
    3: 'Donut',
    4: 'Fries',
    5: 'Hot Dog',
    6: 'Pizza',
    7: 'Sandwich',
    8: 'Taco',
    9: 'Taquito'
}

# Resize parameters
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def is_allowed_file(filename):
    """Check if the file has a valid image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    uploaded_image = None
    upload_feedback = None

    if request.method == 'POST':
        # Save the uploaded file
        file = request.files['file']
        if file and is_allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static', filename)
            file.save(file_path)
            uploaded_image = file_path
            upload_feedback = "" #f"Image '{filename}' uploaded successfully!"

            # Preprocess the image
            img = load_img(file_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = round(np.max(predictions) * 100, 3)  # Convert to percentage
            prediction = class_labels[predicted_class]
        else:
            upload_feedback = "Invalid file. Please upload a valid image (png, jpg, jpeg)."

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        uploaded_image=uploaded_image,
        upload_feedback=upload_feedback
    )


if __name__ == '__main__':
    app.run(debug=True)
