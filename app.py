from flask import Flask, render_template, request
import os
from model import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':

        file = request.files['image']

        if file:

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            file.save(filepath)

            prediction, confidence = predict_image(filepath)

            image_path = filepath

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == '__main__':
    app.run(debug=True)