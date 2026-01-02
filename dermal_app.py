import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import keras
from keras.applications.efficientnet import preprocess_input

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="DermalScan - Face & Skin Condition Predictor", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "DermalScan - AI-Powered Skin Condition Analysis"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content wrapper with glassmorphism */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem 3rem !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
        animation: fadeInDown 0.8s ease-out;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Metrics/Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    /* Subheader styling */
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Path to your trained model file (update if necessary)
MODEL_PATH = "best_dermal_model.h5"
# Default classes order - please set to match train_generator.class_indices order.
DEFAULT_CLASSES = ["clear skin", "dark spot", "puffy eyes", "wrinkles"]

# -----------------------
# CACHING / LOADERS
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load Keras model once and cache it for the app lifetime."""
    model = keras.models.load_model(path)
    return model

@st.cache_resource(show_spinner=False)
def load_haar():
    """Load Haar cascade and cache."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade

# -----------------------
# HELPERS
# -----------------------
def pil_to_cv2(image_pil):
    """Convert PIL.Image to OpenCV BGR numpy array."""
    img = np.array(image_pil)
    # PIL uses RGB, convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv2_to_bytes(img_bgr):
    """Encode BGR image to PNG bytes for Streamlit display."""
    _, im_png = cv2.imencode(".png", img_bgr)
    return im_png.tobytes()

def detect_faces_haar(gray_img, face_cascade, scaleFactor=1.05, minNeighbors=3, minSize=(60,60)):
    """Return list of (x,y,w,h) faces detected by Haar cascade."""
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def predict_on_face(face_bgr, model, classes_list):
    """Preprocess a face crop for EfficientNet, run model, and return (label, conf, raw_probs)."""
    # Resize to model input size
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    # EfficientNet preprocessing
    face_pre = preprocess_input(face_resized.astype("float32"))
    face_input = np.expand_dims(face_pre, axis=0)
    probs = model.predict(face_input)[0]
    idx = int(np.argmax(probs))
    label = classes_list[idx]
    conf = float(probs[idx]) * 100.0
    return label, conf, probs

def draw_annotations(img_bgr, boxes, results, box_color=(0,255,0), thickness=2):
    """Draw rectangles and multi-line labels under/above boxes. Returns annotated image."""
    annotated = img_bgr.copy()
    for (x, y, w, h), res in zip(boxes, results):
        label, conf, probs = res
        # Draw box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), box_color, thickness)

        # Create multi-line label (top) and probabilities (bottom)
        top_text = f"{label} ({conf:.1f}%)"
        # determine where to put top text (avoid going off-image)
        txt_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(annotated, top_text, (x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

    return annotated

def build_probs_overlay(img_bgr, boxes, results, classes_list):
    """Return a small image (numpy array) summarizing class probabilities per detected face to show in sidebar."""
    n = len(results)
    # create white canvas
    w, h = 360, max(100, 40 * n)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    y = 20
    for i, ((x,yb,wb,hb), res) in enumerate(zip(boxes, results)):
        label, conf, probs = res
        header = f"Face {i+1}: {label} ({conf:.1f}%)"
        cv2.putText(canvas, header, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        y += 22
        # per-class probabilities
        for c_idx, cname in enumerate(classes_list):
            p = probs[c_idx] * 100.0
            line = f"  {cname[:18]:18s}: {p:5.1f}%"
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1, cv2.LINE_AA)
            y += 18
        y += 6
    return canvas

# -----------------------
# UI
# -----------------------
# Hero Header
st.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <span style='font-size: 4rem;'>🔬</span>
    </div>
""", unsafe_allow_html=True)

st.title("✨ DermalScan - AI-Powered Skin Analysis")

st.markdown("""
    <div class='subtitle'>
        🎯 Advanced Face Detection & Skin Condition Prediction powered by Deep Learning
    </div>
""", unsafe_allow_html=True)

# Feature highlights
col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)

with col_feat1:
    st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>👤</div>
                <div style='font-weight: 600; margin-top: 0.5rem;'>Face Detection</div>
                <div style='font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem;'>Haar Cascade</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_feat2:
    st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>🧠</div>
                <div style='font-weight: 600; margin-top: 0.5rem;'>Deep Learning</div>
                <div style='font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem;'>EfficientNetB0</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_feat3:
    st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>🎯</div>
                <div style='font-weight: 600; margin-top: 0.5rem;'>4 Conditions</div>
                <div style='font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem;'>High Accuracy</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_feat4:
    st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div style='font-size: 2rem;'>⚡</div>
                <div style='font-weight: 600; margin-top: 0.5rem;'>Real-time</div>
                <div style='font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem;'>Instant Results</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("### ⚙ Settings")
st.sidebar.markdown("---")
model_path_input = st.sidebar.text_input("📁 Model path", value=MODEL_PATH)
classes_input = st.sidebar.text_area(
    "🏷 Class labels (one per line)",
    value="\n".join(DEFAULT_CLASSES),
    help="Enter the classes in the exact numeric order used while training (one per line)."
)
classes = [s.strip() for s in classes_input.splitlines() if s.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚 Detection Settings")
conf_thresh = st.sidebar.slider("📊 Confidence threshold (%)", 0.0, 100.0, 1.0, 0.1)
padding = st.sidebar.slider("📐 Face crop padding (px)", 0, 100, 25, 5)
resize_for_speed = st.sidebar.slider("🖼 Max display width (px)", 300, 1600, 900, 100)

# Load model and cascade (cached)
try:
    with st.spinner('🔄 Loading AI model...'):
        model = load_model(model_path_input)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model at: {model_path_input}\n{e}")
    st.stop()

face_cascade = load_haar()

# Upload section with enhanced styling
st.markdown("### 📤 Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG)", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=False,
    help="Upload a clear frontal face image for best results"
)

if uploaded_file is None:
    st.info("💡 *Getting Started:* Upload an image containing faces to begin the analysis. For optimal results, use clear, well-lit frontal face photos.")
    
    # Add some helpful tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("""
        ✅ *Do:*
        - Use clear, well-lit photos
        - Ensure faces are frontal
        - Use high-resolution images
        - Include the full face
        """)
    
    with tip_col2:
        st.markdown("""
        ❌ *Avoid:*
        - Blurry or dark images
        - Side profiles
        - Covered or obscured faces
        - Very small face sizes
        """)
else:
    # Read image with PIL for reliability, then convert to OpenCV BGR
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_bgr = pil_to_cv2(image_pil)

    # Resize for speed/display if too wide
    orig_h, orig_w = img_bgr.shape[:2]
    scale = 1.0
    if orig_w > resize_for_speed:
        scale = resize_for_speed / orig_w
        img_display = cv2.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)))
    else:
        img_display = img_bgr.copy()

    # Convert to gray and detect faces on the display-sized image for speed
    gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)

    faces = detect_faces_haar(gray, face_cascade, scaleFactor=1.05, minNeighbors=3, minSize=(40,40))

    if len(faces) == 0:
        st.warning("⚠ *No faces detected* - Try a clearer frontal photo or adjust detection settings in the sidebar.")
        st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption="📷 Uploaded image", use_column_width=True)
    else:
        # Processing status
        with st.spinner('🔍 Analyzing faces and detecting skin conditions...'):
            results = []
            boxes_display = []
            for (x, y, w, h) in faces:
                # convert box coords back to original image coordinates
                x0 = int(x / scale)
                y0 = int(y / scale)
                w0 = int(w / scale)
                h0 = int(h / scale)

                # Expand with padding in original image coordinates
                x1 = max(0, x0 - padding)
                y1 = max(0, y0 - padding)
                x2 = min(orig_w, x0 + w0 + padding)
                y2 = min(orig_h, y0 + h0 + padding)

                face_crop = img_bgr[y1:y2, x1:x2].copy()

                # Run prediction
                label, conf, probs = predict_on_face(face_crop, model, classes)
                results.append((label, conf, probs))
                # Add display-scaled box for drawing
                boxes_display.append((int(x), int(y), int(w), int(h)))

        # Filter boxes by confidence threshold (any face has max class >= thresh)
        filtered = []
        filtered_results = []
        filtered_boxes = []
        for box, res in zip(boxes_display, results):
            if res[1] >= conf_thresh:
                filtered.append((box, res))
                filtered_boxes.append(box)
                filtered_results.append(res)

        # If none pass threshold, show all but warn
        if len(filtered_results) == 0:
            st.warning("⚠ All detected faces have confidence below threshold — showing all detections.")
            filtered_boxes = boxes_display
            filtered_results = results

        # Draw annotations on the resized display image
        annotated = draw_annotations(img_display, filtered_boxes, filtered_results, box_color=(0,255,0), thickness=2)

        # Compose probability summary image for sidebar
        probs_canvas = build_probs_overlay(img_display, filtered_boxes, filtered_results, classes)

        # Success message
        st.success(f"✅ *Analysis Complete!* Found {len(filtered_results)} face(s)")
        
        # Display main image and sidebar info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🖼 Analyzed Image")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Annotated with detections")
        with col2:
            st.markdown("### 📊 Detection Results")
            if len(filtered_results) == 0:
                st.write("No detections above threshold.")
            else:
                for i, res in enumerate(filtered_results):
                    lbl, conf, probs = res
                    
                    # Individual face card
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>👤 Face {i+1}</div>
                            <div style='font-size: 1.2rem; font-weight: 600; color: #667eea;'>{lbl}</div>
                            <div style='font-size: 0.9rem; color: #6b7280;'>Confidence: {conf:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # show per-class probabilities as progress bars
                    st.markdown("📈 Probability Breakdown:")
                    for c_name, p_val in zip(classes, probs):
                        st.write(f"{c_name}: {p_val*100:.1f}%")
                        st.progress(float(min(max(p_val, 0.0), 1.0)))
                    
                    if i < len(filtered_results) - 1:
                        st.markdown("---")
            
            st.markdown("### 📋 Summary")
            st.image(cv2.cvtColor(probs_canvas, cv2.COLOR_BGR2RGB), caption="Probability Distribution", use_column_width=True)

        # Provide a download button for annotated image
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        annotated_bytes = cv2_to_bytes(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        st.download_button(
            "⬇ Download Annotated Image", 
            annotated_bytes, 
            file_name="dermalscan_analysis.png", 
            mime="image/png",
            use_container_width=True
        )
        
        st.markdown("""
            <div style='text-align: center; margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 10px;'>
                <div style='font-size: 0.9rem; color: #6b7280;'>
                    <strong>Note:</strong> This is an AI-powered analysis tool. For medical concerns, please consult a healthcare professional.
                </div>
            </div>
        """, unsafe_allow_html=True)
