import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from ultralytics import YOLO

# Page configuration - centered layout for mobile responsiveness
st.set_page_config(
    page_title="Cocoa Disease Detector",
    page_icon="🌱",
    layout="centered"
)

# Inject CSS to make images responsive
st.markdown(
    """
    <style>
    img {
        max-width: 100% !important;
        height: auto !important;
    }
    .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model with caching for fast reloads
def load_model():
    return YOLO('best.pt')

model = st.cache_resource(load_model)()

# Sidebar controls
st.sidebar.title("Settings & Instructions")
st.sidebar.markdown(
    """- Upload a clear, well-lit cocoa leaf or pod image.  
- Adjust the confidence slider to filter detections.  
- Tap 'Run Inference' to view results and download the annotated image."""
)
conf_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01
)
show_original = st.sidebar.checkbox("Show original image", True)

# Main title and description
st.title("🌱 Cocoa Disease Detection")
st.write(
    "Upload an image of a cocoa leaf or pod, then run inference to see detected diseases highlighted with bounding boxes."
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image if enabled
    if show_original:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Run Inference"):
        with st.spinner("Analyzing image..."):
            img_array = np.array(image)
            results = model.predict(img_array, conf=conf_threshold)

        if results:
            result = results[0]
            boxes = result.boxes

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            labels = {0: "0", 1: "1", 2: "2"}

            if boxes:
                # Draw bounding boxes and labels
                for box, cls, conf in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    boxes.conf.cpu().numpy()
                ):
                    x1, y1, x2, y2 = map(int, box)
                    text = f"{labels[int(cls)]}: {conf:.2%}"

                    # Thick white bounding box for clarity
                    draw.rectangle([x1, y1, x2, y2], outline="white", width=4)

                    # Text background and label
                    bbox = draw.textbbox((x1, y1), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
                    draw.text((x1, y1 - text_h), text, fill="white", font=font)

                # Show results
                st.subheader("Inference Results")
                st.image(annotated, caption="Annotated", use_container_width=True)

                # Only show the best (highest-confidence) detection
                cls_arr = boxes.cls.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                best_idx = np.argmax(conf_arr)
                best_label = labels[int(cls_arr[best_idx])]
                best_conf = conf_arr[best_idx]
                st.info(f"Best detection: **{best_label}** with confidence **{best_conf:.2%}**")

                # Download button
                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                st.download_button(
                    "Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="annotated_cocoa.png",
                    mime="image/png"
                )
            else:
                st.warning("No detections found. Try lowering the confidence threshold.")
        else:
            st.error("Inference failed. Please try again.")

# Footer
st.markdown("---")
st.sidebar.info("This app uses a YOLO model trained on cocoa plant images to detect diseases.")
