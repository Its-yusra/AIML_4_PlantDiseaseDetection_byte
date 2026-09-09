import sys
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array


MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"


model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


def predict(image_path):

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    index = np.argmax(predictions)

    label = class_names[index]

    confidence = predictions[index]

    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":

    image_path = sys.argv[1]

    predict(image_path)
