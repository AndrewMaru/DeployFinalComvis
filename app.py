import streamlit as st
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

st.title("Flood Segmentation with U-Net (TFLite)")

@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="unet_flood_segmentation.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

uploaded = st.file_uploader("Upload flood image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((256, 256))
    
    x = np.array(img_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Get input & output tensor info
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])[0, :, :, 0]

    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    st.image(img, caption="Original Image")
    st.image(pred_mask, caption="Predicted Flood Mask", clamp=True)
