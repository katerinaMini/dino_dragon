
import tflite_runtime.interpreter as tflite
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import requests
from PIL import Image
from io import BytesIO
interpreter = tflite.Interpreter(model_path='pred_dino_dragon-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def load_image_from_url(url):
    response = requests.get(url)
    image_data = response.content
    img = Image.open(BytesIO(image_data))
    img = img.resize((150, 150))  
    return img

def preprocess_input(x):
    x = np.array(x, dtype='float32')
    X = np.array([x])
    X /= 255
    return X

def predict(img):
    X = preprocess_input(img)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    if preds[0] < 0.5:
        return "Модель предсказывает класс 0 (дино)"
    else:
        return "Модель предсказывает класс 1 (дракон)"


def lambda_handler(event, context):
        img_url = event['url']
        img = load_image_from_url(img_url)
        preds = predict(img)
        return {
            "StatusCode": 200,
            "body": preds
        }

