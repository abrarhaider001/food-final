from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module
import tensorflow as tf
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This allows all origins

cnn = tf.keras.models.load_model('trained_model.h5')

class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
               'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
               'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
               'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
               'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    try:
        if request.method == "OPTIONS":
            return jsonify({"message": "CORS preflight request"}), 200  # Preflight response

        # Get JSON data from request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 string to image
        image_data = base64.b64decode(data['image'])
        image = tf.keras.preprocessing.image.load_img(BytesIO(image_data), target_size=(64, 64))

        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)

        # Make prediction
        predictions = cnn.predict(input_arr)
        result_index = np.argmax(predictions)

        return jsonify({"prediction": class_names[result_index]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
