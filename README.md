# ✨ DermalScan - AI-Powered Skin Analysis

![DermalScan Badge](https://img.shields.io/badge/AI-Deep%20Learning-blue)
![Python Badge](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Streamlit Badge](https://img.shields.io/badge/Streamlit-App-red)

DermalScan is an advanced AI-powered web application designed to analyze facial skin conditions in real-time. Leveraging deep learning (EfficientNetB0) and computer vision (OpenCV), it detects faces and predicts skin conditions such as "clear skin", "dark spot", "puffy eyes", and "wrinkles" with high accuracy.

## 🚀 Key Features

-   **🔍 Automatic Face Detection**: Uses Haar Casecade classifiers to instantly locate faces in uploaded images.
-   **🧠 Deep Learning Analysis**: Powered by a fine-tuned EfficientNetB0 model for robust skin condition classification.
-   **⚡ Real-time Results**: Instant processing and visualization of analysis on the image.
-   **📊 Detailed Probabilities**: View confidence scores for all predicted classes.
-   **💾 Export Capability**: Download the annotated image with analysis results overlaid.
-   **🎨 Modern UI**: A sleek, user-friendly interface built with Streamlit.

---

## 🛠 Tech Stack

-   **Frontend**: Streamlit
-   **Computer Vision**: OpenCV (Face detection, image annotation)
-   **Deep Learning**: TensorFlow / Keras (EfficientNetB0 model)
-   **Image Processing**: NumPy, PIL

---

## 📦 Installation & Setup

### Prerequisites

-   Python 3.8 or higher installed.
-   Git (optional, for cloning).

### 1. Clone the Repository (or Download Source)

```bash
git clone <repository-url>
cd dermalscan
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have the model file `best_dermal_model.h5` in the root directory. If it's missing, you may need to train the model using `accuracy.py` or download the pre-trained weights.

---

## 📖 User Guide

1.  **Run the Application**:
    Execute the following command in your terminal:
    ```bash
    streamlit run dermal_app.py
    ```

2.  **Upload an Image**:
    -   Click the "Browse files" button in the app.
    -   Select a clear, frontal face image (JPEG/PNG).

3.  **View Results**:
    -   The app will process the image and draw boxes around detected faces.
    -   Conditions and confidence scores will be displayed on the image and in the sidebar.

4.  **Download**:
    -   Click "Download Annotated Image" to save the result.

5.  **Settings (Sidebar)**:
    -   **Confidence Threshold**: Adjust to filter out low-confidence predictions.
    -   **Face Crop Padding**: Change how tight the crop around the face is for the model.
    -   **Max Display Width**: Adjust image size for performance.

---

## 💻 Developer Guide

### Project Structure

-   **`dermal_app.py`**: The main entry point. Contains the Streamlit UI code, image processing logic, and inference pipeline.
-   **`accuracy.py`**: Script used for data preparation, augmentation, and training the EfficientNetB0 model.
-   **`best_dermal_model.h5`**: The saved Keras model file (required for inference).
-   **`requirements.txt`**: List of Python dependencies.

### Key Functions (`dermal_app.py`)

-   `load_model(path)`: Caches and loads the Keras model.
-   `detect_faces_haar(...)`: Uses OpenCV's Haar Cascade to find face coordinates.
-   `predict_on_face(...)`: Preprocesses the face crop and runs it through the model.
-   `draw_annotations(...)`: Visualizes the bounding boxes and labels on the image.

### Extending the Model

To train on new data:
1.  Organize your dataset into folders by class (e.g., `dataset/class1`, `dataset/class2`).
2.  Modify `accuracy.py` to point to your new dataset path.
3.  Run `accuracy.py` to retrain and generate a new `best_dermal_model.h5`.
4.  Update `DEFAULT_CLASSES` in `dermal_app.py` to match your new class labels.

---

## 🤝 Contribution

Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License

[MIT License](LICENSE)
