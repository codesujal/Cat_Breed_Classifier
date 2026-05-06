import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("catbreed_model.keras")

breed_names = {
    1: "Abyssinian",
    2: "Bengal",
    3: "Birman",
    4: "Bombay",
    5: "British Shorthair",
    6: "Egyptian Mau",
    7: "Maine Coon",
    8: "Persian",
    9: "Ragdoll",
    10: "Russian Blue",
    11: "Siamese",
    12: "Sphynx"
}

st.title("Cat Breed Classifier")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]

    top_3_idx = np.argsort(preds)[-3:][::-1]

    st.subheader("Top 3 Predictions")

    for i in top_3_idx:
        breed_name = breed_names.get(i + 1, "Unknown")
        confidence = preds[i] * 100

        st.write(f"{breed_name}: {confidence:.2f}%")