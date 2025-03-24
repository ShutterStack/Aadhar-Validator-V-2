import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

# Load models
yolo_model = YOLO("best.pt")  # YOLOv8 model for cropping/detection
tf_model = load_model("Card_Detector_ResNet.keras")  # TensorFlow model for classification

# Labels for binary classification
labels_dict = {1: 'Aadhar Card', 0: 'Not an Aadhar Card'}

# Function to adjust orientation
def adjust_orientation(img):
    """Check and rotate image to horizontal if vertical."""
    height, width = img.shape[:2]
    if height > width:  # Vertical image
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

# Function to preprocess image for TensorFlow model
def preprocess_for_model(image):
    """Convert image to grayscale, resize, normalize, and reshape for model input."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, (100, 100))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 100, 100, 1)
    return reshaped

# Function to process image: crop with YOLOv8 and predict with TensorFlow
def process_image(image):
    # Convert PIL Image to OpenCV format (numpy array)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Adjust orientation to horizontal
    img = adjust_orientation(img)

    # Step 1: Detect regions with YOLOv8
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Format: [x_min, y_min, x_max, y_max]

    # If no region is detected, use the entire image for verification
    if len(boxes) == 0:
        st.warning("No regions detected! Using the full image for verification.")
        cropped_result = img
    else:
        # Determine the bounding box that encloses all detected regions
        x_min = int(np.min(boxes[:, 0]))
        y_min = int(np.min(boxes[:, 1]))
        x_max = int(np.max(boxes[:, 2]))
        y_max = int(np.max(boxes[:, 3]))
        cropped_result = img[y_min:y_max, x_min:x_max]
        st.write(f"Detected region cropped: ({x_min}, {y_min}) to ({x_max}, {y_max})")

    # Create augmented versions: original, vertical flip, horizontal flip, and both axes flipped
    images = {
        "original": cropped_result,
        "flip_vertical": cv2.flip(cropped_result, 0),
        "flip_horizontal": cv2.flip(cropped_result, 1),
        "flip_both": cv2.flip(cropped_result, -1)
    }

    predictions = []
    # Process each augmented version and get the model's prediction
    for key, im in images.items():
        processed = preprocess_for_model(im)
        pred = tf_model.predict(processed, verbose=0)[0][0]  # Assuming model outputs a probability
        predictions.append(pred)

    # Average the prediction scores
    avg_score = np.mean(predictions)
    label = 1 if avg_score >= 0.75 else 0  # Using threshold of 0.75
    confidence = avg_score * 100 if label == 1 else (1 - avg_score) * 100
    output_text = f"{labels_dict[label]} (Average Confidence: {confidence:.2f}%)"

    # Convert cropped result to RGB for display
    cropped_result = cv2.cvtColor(cropped_result, cv2.COLOR_BGR2RGB)
    return cropped_result, output_text

# Streamlit app
st.title("Aadhar Card Crop & Predict Tool")
st.write("Upload an image to crop and predict if it's an Aadhar card.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", width=500)

    # Process the image
    with st.spinner("Processing image..."):
        cropped_img, result_text = process_image(image)

    # Display the cropped image and prediction result
    st.image(cropped_img, caption="Cropped Image", width=500)
    st.markdown(f"**Prediction Result:** {result_text}")

    # Option to download the cropped image
    img_pil = Image.fromarray(cropped_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Cropped Image",
        data=byte_im,
        file_name="cropped_aadhar_card.jpg",
        mime="image/jpeg"
    )