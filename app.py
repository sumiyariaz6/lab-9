import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Object Detection App")

st.title("🔍 Object Detection using YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            results = model(np.array(image))

            annotated_image = results[0].plot()

            st.image(annotated_image, caption="Detected Objects", use_container_width=True)

            st.success("Detection Completed!")
