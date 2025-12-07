# app.py (Streamlit)
import streamlit as st
from huggingface_hub import hf_hub_download
import zipfile, os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

REPO_ID = "AndrewMaru/unet-flood-segmentation"
ZIP_NAME = "unet_savedmodel.zip"

@st.cache_resource
def download_and_load_model():
    # download zip from hf hub (cached locally by huggingface_hub)
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME)
    extract_dir = "/tmp/unet_savedmodel"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
    # load TF SavedModel (no compile)
    model = tf.keras.models.load_model(extract_dir, compile=False)
    return model

model = download_and_load_model()

st.title("Flood Segmentation (HF-hosted model)")

uploaded = st.file_uploader("Upload image", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((256,256))
    x = np.array(img_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0,:,:,0]
    mask = (pred > 0.5).astype("uint8") * 255
    st.image(img, caption="Original")
    st.image(mask, caption="Predicted Mask", clamp=True)
