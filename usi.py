import streamlit as st
import numpy as np
import cv2
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
import shap
from lime import lime_image
from skimage.color import rgb2gray
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

CNN_MODEL_PATH = "IRM_CNN.h5"
RF_MODEL_PATH = "RF_model.pkl"
SVM_MODEL_PATH = "MRI_SVM_model.pkl"
DATA_PATH = "brisc2025_balanced_aug/train"

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CNN_IMG_SIZE = (50, 50)
HOG_IMG_SIZE = (128, 128)
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

def preprocess_for_cnn(img_bgr):
    img = cv2.resize(img_bgr, CNN_IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gray.astype('float32') / 255.0
    return gray.reshape(CNN_IMG_SIZE[1], CNN_IMG_SIZE[0], 1)

def preprocess_for_rf(img_bgr):
    img = cv2.resize(img_bgr, CNN_IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gray.astype('float32') / 255.0
    return gray.flatten()

def preprocess_for_svm(img_bgr):
    img = cv2.resize(img_bgr, HOG_IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        feature_vector=True
    )
    return features

@st.cache_resource
def load_models():
    cnn = load_model(CNN_MODEL_PATH)
   
    input_shape = (None, CNN_IMG_SIZE[1], CNN_IMG_SIZE[0], 1)
    cnn.build(input_shape=input_shape)
    dummy_input = np.zeros((1, CNN_IMG_SIZE[1], CNN_IMG_SIZE[0], 1), dtype=np.float32)
    _ = cnn.predict(dummy_input, verbose=0)
    
    with open(RF_MODEL_PATH, "rb") as f:
        rf = pickle.load(f)
    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)
    return cnn, rf, svm

cnn_model, rf_model, svm_model = load_models()

if not hasattr(svm_model, "predict_proba"):
    st.warning("SVM model lacks probability=True; soft confidence unavailable.")

def sample_background_images(n=50):
    imgs = []
    for cname in CLASS_NAMES:
        folder = os.path.join(DATA_PATH, cname)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        for fname in files[: max(1, n // len(CLASS_NAMES))]:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)
    return imgs

BG_IMAGES = sample_background_images(50)
def make_gradcam_heatmap_cnn(model, img_array, last_conv_layer_name=None, pred_index=None):
    if not model.built:
        dummy_input = np.zeros_like(img_array)
        _ = model.predict(dummy_input, verbose=0) 

    last_conv_layer_name = 'conv4' 

    last_conv_layer = model.get_layer(last_conv_layer_name)
    output_layer = model.get_layer('output') 
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, output_layer.output] 
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)


        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        print("ERROR: Gradients are None. Grad-CAM cannot be computed.")
        return np.zeros(CNN_IMG_SIZE)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
        
    heatmap = cv2.resize(heatmap, CNN_IMG_SIZE)
    return heatmap

def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.45):
    img_resized = cv2.resize(orig_bgr, CNN_IMG_SIZE)
    heat_uint = np.uint8(255 * heatmap)
    heat_colored = cv2.applyColorMap(heat_uint, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heat_colored, alpha, 0)
    return overlay

@st.cache_resource
def make_rf_explainer(bg_imgs=None):
    if not bg_imgs:
        background = np.zeros((1, CNN_IMG_SIZE[0]*CNN_IMG_SIZE[1]))
    else:
        feats = [preprocess_for_rf(img) for img in bg_imgs[:50]]
        background = np.array(feats)
    return shap.TreeExplainer(rf_model, background)

rf_explainer = make_rf_explainer(BG_IMAGES)

def svm_predict_hog_batch(images_rgb):
    probs = []
    for img in images_rgb:
        bgr = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
        feat = preprocess_for_svm(bgr).reshape(1, -1)
        probs.append(svm_model.predict_proba(feat)[0])
    return np.array(probs)

st.title("🧠 MRI Brain Tumor Classification — CNN + RF + SVM with Explainability")
st.write("Upload an MRI image to classify and visualize explanations for each model.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to continue.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
st.image(pil_img, caption="Input Image", use_column_width=True)

cnn_input = preprocess_for_cnn(img_bgr)[None, ...]
cnn_probs = cnn_model.predict(cnn_input, verbose=0)[0]
cnn_pred = int(np.argmax(cnn_probs))
cnn_conf = float(cnn_probs[cnn_pred])

rf_feat = preprocess_for_rf(img_bgr).reshape(1, -1)
rf_probs = rf_model.predict_proba(rf_feat)[0]
rf_pred = int(np.argmax(rf_probs))
rf_conf = float(rf_probs[rf_pred])

svm_feat = preprocess_for_svm(img_bgr).reshape(1, -1)
if hasattr(svm_model, "predict_proba"):
    svm_probs = svm_model.predict_proba(svm_feat)[0]
    svm_pred = int(np.argmax(svm_probs))
    svm_conf = float(svm_probs[svm_pred])
else:
    svm_pred = int(svm_model.predict(svm_feat))
    svm_conf = None

# Display individual model results
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### CNN")
    st.write(f"**Pred:** {CLASS_NAMES[cnn_pred]} \n**Conf:** {cnn_conf:.3f}")
with col2:
    st.markdown("### Random Forest")
    st.write(f"**Pred:** {CLASS_NAMES[rf_pred]} \n**Conf:** {rf_conf:.3f}")
with col3:
    st.markdown("### SVM")
    st.write(f"**Pred:** {CLASS_NAMES[svm_pred]} \n**Conf:** {svm_conf if svm_conf else 'N/A'}")

st.markdown("---")

predictions = [cnn_pred, rf_pred]
if svm_conf is not None:
    predictions.append(svm_pred)
    
winning_pred_index = None
if len(predictions) > 0:
    from collections import Counter
    vote_counts = Counter(predictions)
    final_pred_index = vote_counts.most_common(1)[0][0]
    winning_models = [
        "CNN" if cnn_pred == final_pred_index else None,
        "Random Forest" if rf_pred == final_pred_index else None
    ]
    if svm_conf is not None:
        winning_models.append("SVM" if svm_pred == final_pred_index else None)
    
    winning_models = [m for m in winning_models if m is not None]
    
    st.subheader(f"🧠 Final Consensus Prediction: **{CLASS_NAMES[final_pred_index]}**")
    st.success(f"The model consensus is **{CLASS_NAMES[final_pred_index]}** (Votes: {vote_counts[final_pred_index]} of {len(predictions)}). Winning models: {', '.join(winning_models)}")
    
    winning_pred_index = final_pred_index

st.markdown("---")

st.subheader("✨ Explainability for Consensus Model(s)")

if winning_pred_index is not None and cnn_pred == winning_pred_index:
    st.markdown("### 1️⃣ CNN — Grad-CAM")
    heatmap = make_gradcam_heatmap_cnn(cnn_model, cnn_input)
    overlay = overlay_heatmap_on_image(img_bgr, heatmap)
    
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
              caption=f"Grad-CAM (Pred: {CLASS_NAMES[cnn_pred]} | Conf: {cnn_conf:.3f})",
              use_column_width=True)

    fig_cam, ax_cam = plt.subplots()
    ax_cam.imshow(heatmap, cmap="jet")
    ax_cam.axis("off")
    st.pyplot(fig_cam)
    
    st.markdown("---")

if winning_pred_index is not None and rf_pred == winning_pred_index:
    st.markdown("### 2️⃣ Random Forest — SHAP")
    rf_shap_vals = rf_explainer.shap_values(rf_feat)

    if isinstance(rf_shap_vals, list):
        if rf_pred < len(rf_shap_vals):
            shap_values_for_pred = rf_shap_vals[rf_pred]
        else:
            st.warning(f"RF SHAP output size is ambiguous. Using the only available index [0].")
            shap_values_for_pred = rf_shap_vals[0]
    else:
        shap_values_for_pred = rf_shap_vals
        
    if shap_values_for_pred.ndim > 1:
        shap_values_for_pred = shap_values_for_pred.flatten()
    if shap_values_for_pred.size == 10000:
        SHAP_VIS_SIZE = (100, 100)
    else:
        SHAP_VIS_SIZE = (CNN_IMG_SIZE[1], CNN_IMG_SIZE[0])

    shap_img = shap_values_for_pred.reshape(SHAP_VIS_SIZE)

    if SHAP_VIS_SIZE != (CNN_IMG_SIZE[1], CNN_IMG_SIZE[0]):
        shap_img_resized = cv2.resize(shap_img, CNN_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    else:
        shap_img_resized = shap_img

    shap_norm = (shap_img_resized - np.min(shap_img_resized)) / (np.ptp(shap_img_resized) + 1e-6)
    color = cv2.applyColorMap(np.uint8(255 * shap_norm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(img_bgr, CNN_IMG_SIZE), 0.6, color, 0.4, 0)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True,caption=f"RF SHAP Overlay — {CLASS_NAMES[rf_pred]}")


    fig_rf, ax_rf = plt.subplots()
    ax_rf.imshow(shap_norm, cmap="seismic")
    ax_rf.axis("off")
    st.pyplot(fig_rf)
    
    st.markdown("---")


if winning_pred_index is not None and svm_conf is not None and svm_pred == winning_pred_index:
    st.markdown("### 3️⃣ SVM — LIME (on HOG features)")
    lime_exp = lime_image.LimeImageExplainer()
    with st.spinner("Computing LIME explanation..."):
        explanation = lime_exp.explain_instance(
            img_rgb.astype("double"),
            svm_predict_hog_batch,
            top_labels=3,
            hide_color=0,
            num_samples=300
        )

    temp, mask = explanation.get_image_and_mask(
        label=svm_pred,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    temp_display = (temp * 255).astype(np.uint8)

    st.image(temp_display, caption=f"LIME — {CLASS_NAMES[svm_pred]}", use_column_width=True)

    

if winning_pred_index is None or (cnn_pred != winning_pred_index and rf_pred != winning_pred_index and (svm_conf is None or svm_pred != winning_pred_index)):
    st.info("No explainability visuals are displayed because no model achieved the final consensus prediction.")

st.success("✅ Done — The interface is fully operational, displaying predictions and conditional explanations.")