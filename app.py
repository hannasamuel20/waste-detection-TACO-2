import streamlit as st
import numpy as np
import tempfile
import cv2
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

st.set_page_config(page_title="TACO Litter Detection", layout="centered")
st.title("🗑️ TACO Waste Detection")

# File uploader
file_type = st.selectbox("Choose input type", ["Image", "Video", "Webcam"])

# ---- IMAGE HANDLING ----
if file_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting..."):
                image_np = np.array(image)
                results = model.predict(source=image_np, conf=0.25)
                result_img = results[0].plot()
                st.image(result_img, caption="Detection Results", use_container_width=True)

# ---- VIDEO HANDLING ----
elif file_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.button("Run Detection"):
            stframe = st.empty()
            cap = cv2.VideoCapture(tfile.name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO detection
                results = model.predict(source=frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()

                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()

# ---- REALTIME WEBCAM DETECTION ----
elif file_type == "Webcam":
    st.warning("Webcam support is limited in Streamlit cloud/web. For real-time webcam detection, run locally.")

    if st.button("Start Webcam Detection"):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)  # Use your webcam (device index 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
