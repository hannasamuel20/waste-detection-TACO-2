import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np


# Load the trained model
model = YOLO("notebooks/yolov8m.pt")

st.set_page_config(page_title="TACO Litter Detection", layout="centered")
st.title("🗑️ TACO Waste Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            # Convert PIL to NumPy
            image_np = np.array(image)

            # Run inference
            results = model.predict(source=image_np, conf=0.25)

            # Get result image
            result_img = results[0].plot()

            st.image(result_img, caption="Detection Results", use_column_width=True)
