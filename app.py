import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IdentityShield", # Clean title
    page_icon="🔒",
    layout="centered"
)

# --- 2. CONNECTING THE CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# --- 3. THE AI ENGINE (Keep your logic exactly the same) ---
def process_image(image_file, blur_strength, shape):
    # ... (Keep this function EXACTLY as it was in your previous code) ...
    # Copy-paste the 'process_image' function from your old file here
    # If you need me to paste it again, let me know, but try to keep your logic.
    
    # Just to be safe, here is the Logic block again:
    image = Image.open(image_file)
    image_np = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    processed_img = image_np.copy()
    
    for (x, y, w, h) in faces:
        roi = processed_img[y:y+h, x:x+w]
        k = int(blur_strength)
        if k % 2 == 0: k += 1
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
        
        if shape == "Circle":
            mask = np.zeros((h, w), dtype="uint8")
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            cv2.circle(mask, center, radius, 255, -1)
            processed_img[y:y+h, x:x+w] = np.where(mask[..., None] == 255, blurred_roi, roi)
        else:
            processed_img[y:y+h, x:x+w] = blurred_roi
            
        cv2.rectangle(processed_img, (x, y), (x+w, y+h), (99, 102, 241), 2) # Changed border to Indigo

    return processed_img, len(faces)


# --- 4. THE NEW UI (Custom Logo) ---

# Use HTML to create a custom "Tech Logo" instead of Streamlit's default title
st.markdown("""
    <div>
        <span class="logo-text">IdentityShield</span>
        <span class="logo-badge">PRO v2.0</span>
    </div>
""", unsafe_allow_html=True)

st.markdown('<p style="color:#94a3b8; margin-bottom: 30px;">Enterprise-grade biometric anonymization for privacy compliance.</p>', unsafe_allow_html=True)

# Container for controls (Glass Card)
st.markdown('<div class="css-card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    blur_amount = st.slider("Blur Intensity", 1, 99, 61)
with col2:
    blur_shape = st.radio("Blur Shape", ["Square", "Circle"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("Upload Image to Anonymize", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    if st.button("🔒 Secure Image"):
        with st.spinner("Encrypting visual data..."):
            result_img, face_count = process_image(uploaded_file, blur_amount, blur_shape)
            
            st.success(f"Security Protocol Complete: {face_count} identities protected.")
            st.image(result_img, caption="Secured output", use_column_width=True)