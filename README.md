# 🛡️ IdentityShield Pro



**IdentityShield Pro** is a privacy-focused computer vision application designed to automatically detect and anonymize biometric data (faces) in images. Built for enterprise compliance and data protection.

---

## 🚀 Key Features

* **Autonomic Face Detection:** Uses Haar Cascade classifiers to identify facial features in real-time.
* **Dynamic Privacy Masking:** Apply Gaussian Blur with adjustable intensity to detected regions.
* **Smart Masking Shapes:** Toggle between standard rectangular masking or circular masking using NumPy bitwise operations.
* **Zero-Data Retention:** Processing happens locally in memory; no images are stored on any server.

## 🛠️ Tech Stack

* **Logic:** Python 3.11
* **Computer Vision:** OpenCV (cv2), NumPy
* **Frontend:** Streamlit, CSS3
* **Image Processing:** Pillow (PIL)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YourUsername/IdentityShield-Pro.git](https://github.com/YourUsername/IdentityShield-Pro.git)
    cd IdentityShield-Pro
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python -m streamlit run app.py
    ```

## 🧠 How It Works

1.  **Input:** The user uploads an image (`.jpg`, `.png`).
2.  **Preprocessing:** The image is converted to a NumPy array and then to Grayscale (`cv2.COLOR_RGB2GRAY`) to improve detection accuracy.
3.  **Detection:** The **Haar Cascade** algorithm scans the image for Haar-like features (edges, lines) to locate faces.
4.  **Processing:**
    * A Region of Interest (ROI) is extracted for each face.
    * A **Gaussian Blur** kernel `(k, k)` is applied to the ROI.
    * If "Circle" mode is selected, a binary mask is generated to blend the blurred ROI with the original image seamlessly.
5.  **Output:** The processed image is rendered back to the UI.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

**Developed by Zakaria El Alami**
*Ibn Tofail University - 3rd Year AI Student*
