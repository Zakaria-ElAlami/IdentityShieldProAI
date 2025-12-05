import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. PAGE CONFIG (Must be first) ---
st.set_page_config(
    page_title="IdentityShield Pro",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. INTEGRATE CSS (The Bridge) ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the external CSS file
local_css("style.css")

# --- 3. THE PYTHON LOGIC (Backend) ---
def detect_and_blur(image_file, blur_strength, blur_shape):
    # Convert image
    image = Image.open(image_file)
    image_np = np.array(image)
    
    # Load Detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Process
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    annotated_image = image_np.copy()
    
    for (x, y, w, h) in faces:
        # Create ROI (Region of Interest)
        roi = annotated_image[y:y+h, x:x+w]
        
        # Apply Blur
        k = blur_strength
        if k % 2 == 0: k += 1
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
        
        # Shape Logic (Circle vs Square)
        if blur_shape == "Circle":
            # Create a circular mask
            mask = np.zeros((h, w), dtype="uint8")
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            cv2.circle(mask, center, radius, 255, -1)
            
            # Combine blurred face with original using mask
            # Where mask is white, use blurred. Where black, use original.
            annotated_image[y:y+h, x:x+w] = np.where(
                mask[..., None] == 255, blurred_roi, roi
            )
        else:
            # Standard Square Blur
            annotated_image[y:y+h, x:x+w] = blurred_roi
            
        # Draw sleek border
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 157), 2)

    return annotated_image, len(faces)

# --- 4. THE UI (Frontend Structure) ---
st.title("🛡️ IdentityShield Pro")
st.markdown("### The Enterprise Privacy Tool")

# Layout using Columns (Responsivity)
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.info("👇 **Settings**")
    blur_amount = st.slider("Blur Intensity", 1, 99, 51)
    blur_shape = st.radio("Blur Shape", ["Square", "Circle"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

if uploaded_file:
    st.write("---")
    if st.button("🔒 Anonymize Identity"):
        with st.spinner("Processing biometric data..."):
            result, count = detect_and_blur(uploaded_file, blur_amount, blur_shape)
            
            st.success(f"Security Protocol Active: {count} faces anonymized.")
            st.image(result, caption="Processed Image", use_column_width=True)