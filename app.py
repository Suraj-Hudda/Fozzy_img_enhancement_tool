from flask import Flask, render_template, request, send_file
from models.stepcode1 import grayscale_enhance
from models.stepcode2 import rgb_enhance
import cv2
import numpy as np
import io

app = Flask(__name__)

def grayscale_enhancement(image):
    enhance=grayscale_enhance(image)
    return cv2.cvtColor(cv2.cvtColor(enhance, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    

def rgb_enhancement(image):
    enhance=rgb_enhance(image)
    return enhance

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        enhancement_type = request.form['enhancement_type']
        
        # Read the image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if enhancement_type == 'grayscale':
            enhanced_image = grayscale_enhancement(image)
        else:  # RGB
            enhanced_image = rgb_enhancement(image)
        
        # Convert the enhanced image to bytes
        is_success, buffer = cv2.imencode(".jpg", enhanced_image)
        io_buf = io.BytesIO(buffer)
        
        return send_file(io_buf, mimetype='image/jpeg')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)