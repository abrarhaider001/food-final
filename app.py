import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from io import BytesIO  # To read image file

app = Flask(__name__) # Create Flask app

cnn = tf.keras.models.load_model('trained_model.h5')

class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
               'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
               'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
               'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
               'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = tf.keras.preprocessing.image.load_img(BytesIO(file.read()), target_size=(64, 64))

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)

    print(predictions)

    result_index = np.argmax(predictions) #Return index of max element
    print(result_index)
    print("It's a {}".format(class_names[result_index]))
    return jsonify({"prediction": class_names[result_index]})

if __name__ == '__main__':
    app.run(debug=True)