import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model_path = "model/food_freshness_model.keras"
model = load_model(model_path)

class_names = ["fresh apples", "fresh banana", "fresh oranges",
               "rotten apples", "rotten banana", "rotten oranges"]

st.title("🍎 Food Freshness Detection")
st.write("Upload an image of fruit to predict its freshness.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred)) * 100

    st.markdown(f"### 🔍 Prediction: **{pred_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")
