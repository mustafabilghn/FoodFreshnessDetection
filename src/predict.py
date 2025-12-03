from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = "model/food_freshness_model.keras"
model = load_model(model_path)

class_names = [
    "fresh apples",
    "fresh banana",
    "fresh oranges",
    "rotten apples",
    "rotten banana",
    "rotten oranges"
]

image_path = "../data/freshapples/rotated_by_15_Screen Shot 2018-06-08 at 5.00.03 PM.png"

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]

print(f"Image: {os.path.basename(image_path)} --> Prediction: {pred_class}")
