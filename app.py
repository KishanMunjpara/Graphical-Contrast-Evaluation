from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired  # Import InputRequired validator
from werkzeug.utils import secure_filename
import cv2
import imutils
import base64

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

class UploadFileForm(FlaskForm):
    file1 = FileField("Image 1", validators=[InputRequired()])
    file2 = FileField("Image 2", validators=[InputRequired()])
    submit = SubmitField("Compare Images")

def calculate_similarity(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compare images
    similar, _ = compare_ssim(gray1, gray2, full=True)
    return similar

def compare_images(image1, image2):
    # Read and resize images
    img1 = cv2.imdecode(np.fromstring(image1.read(), np.uint8), cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1, (600, 800))
    img2 = cv2.imdecode(np.fromstring(image2.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.resize(img2, (600, 800))
    
    # Calculate similarity
    similarity = calculate_similarity(img1, img2)
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compare images
    similar, diff = compare_ssim(gray1, gray2, full=True)
    
    # Normalize the difference image
    diff = (diff * 255).astype(np.uint8)
    
    # Check if similarity is higher than a certain threshold
    if similar > 0.9:  # Change the threshold value as needed
        result_text = "Images are similar"
    else:
        result_text = "Images are different"
    
    # Threshold the difference image to create a mask
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Draw bounding boxes around the differing areas on the second image only
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Draw black bounding box only on img2
    
    # Set margin color to black
    margin = np.zeros((img1.shape[0], 50, 3), dtype=np.uint8)
    margin[:, :, :] = (0, 0, 0)  # Set margin color to black
    
    # Combine images with margin between them
    combined_img = np.hstack((img1, margin, img2))
    
    # Add similarity score text below the images
    cv2.putText(combined_img, "Similarity: {:.2f}".format(similarity), (int(combined_img.shape[1] / 2) - 70, img1.shape[0] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Encode combined image to base64
    retval, buffer = cv2.imencode('.jpg', combined_img)
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return result_base64, similarity

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    result_image = None
    similarity_value = None
    if form.validate_on_submit():
        image1 = request.files['file1']
        image2 = request.files['file2']
        result_image, similarity_value = compare_images(image1, image2)
    return render_template('index.html', form=form, result_image=result_image, similarity=similarity_value)


