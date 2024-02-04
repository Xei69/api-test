import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
import time
import os

import logging
from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
KEY = os.environ['KEY']
INTERPRETER_POOL_SIZE = 5  # Number of interpreters in the pool

# Load the labels
labels = eval(os.environ["POKEMONS"])

def preprocess_image(image):
    img = tf.image.resize(image, [224, 224])
    img = img / 255.0  # Normalize input data
    return img

def predict_pokemon_from_image(interpreter, image):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor value
    input_data = tf.convert_to_tensor(image, dtype=tf.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the prediction output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_label = labels[predicted_class_index]

    return predicted_label

def predict_pokemon_from_url(image_url):
    # Download the image data from the URL
    response = requests.get(image_url)
    image_data = response.content

    # Decode the image data as a tensor
    image = tf.image.decode_image(image_data, channels=3)
    image = preprocess_image(image)

    # Load the TFLite model from the interpreter pool
    interpreter = get_interpreter_from_pool()

    predicted_pokemon = predict_pokemon_from_image(interpreter, [image.numpy()])

    # Return the interpreter back to the pool
    return_interpreter_to_pool(interpreter)

    return predicted_pokemon

def initialize_interpreter():
    # Load the TFLite model
    tflite_model_path = 'pokefier_t1.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_interpreter_from_pool():
    return interpreter_pool.pop()

def return_interpreter_to_pool(interpreter):
    interpreter_pool.append(interpreter)

@app.route('/', methods=['GET'])
def handle_get_request():
    app.logger.info('GET request received')
    return 'Hello, World!'

@app.route('/identifyPokemon', methods=['POST'])
def identify_pokemon():
    received_data = request.json

    if received_data['key'] != KEY:
        return ''

    pokemon_image = received_data['image']
    pokemon_name = predict_pokemon_from_url(pokemon_image)

    return pokemon_name

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.logger.info('Server is starting...')
    # Initialize the interpreter pool
    interpreter_pool = [initialize_interpreter() for _ in range(INTERPRETER_POOL_SIZE)]
    # Start the Flask server
  
    app.run(host='0.0.0.0', debug=True)
