from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
app = Flask(__name__)

# Define and load the model
model_unet = tf.keras.models.load_model('model_segmentationUnet.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    # Get the uploaded image file
    image_file = request.files['file']

    # Open the image and convert it to RGB
    img = Image.open(image_file)
    img_rgb = img.convert('RGB')

    # Resize the image to match the input shape of the model
    img_resized = img_rgb.resize((256, 256))

    # Convert the image to a numpy array
    img_array = np.array(img_resized) / 255.0  # Normalize pixel values

    # Perform segmentation using the model
    segmentation = model_unet.predict(np.expand_dims(img_array, axis=0))
    print("hello" , segmentation)
    # Create an image from the segmentation array
    segmentation_array_scaled = (segmentation * 255).astype(np.uint8)
    segmentation_array_reshaped = np.squeeze(segmentation_array_scaled, axis=(0, 3))
    segmentation_image = Image.fromarray(segmentation_array_reshaped)

    # 
    plt.imshow(segmentation_array_reshaped, cmap='gnuplot', alpha=1.0)
    plt.axis('off')
    # 
    output_dir = os.path.join(app.root_path, 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'segmentation_output.jpg')
    # segmentation_image.save(output_file_path)
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    print("Segmentation output image saved at:", output_file_path)
    plt.close()
    return render_template('result.html', segmentation_path='static/images/segmentation_output.jpg')

if __name__ == '__main__':
    app.run(debug=True)
