import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models once at module load time
yolo_model = YOLO("best.pt")  # YOLOv8 model for detecting the region
tf_model = load_model("Card_Detector_ResNet.keras")  # TensorFlow model for classification
THRESHOLD = 0.75  # Prediction threshold for Aadhar Card

def adjust_orientation(img):
    """Rotate image 90° clockwise if it's vertical."""
    height, width = img.shape[:2]
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def preprocess_for_model(image):
    """
    Convert image to grayscale, resize to 100x100,
    normalize the pixel values, and reshape for model input.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, (100, 100))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 100, 100, 1)
    return reshaped

def aadhar(image_path: str) -> bool:
    """
    Given an image path, this function:
      - Loads and adjusts the image orientation,
      - Detects regions using YOLOv8,
      - Crops the image to the ROI (enclosing all detections),
      - Creates flipped augmentations (vertical, horizontal, both),
      - Processes each augmented version through the TensorFlow model,
      - Averages the predictions,
      - Returns True if the average prediction is >= THRESHOLD,
        else returns False.
    If no region is detected, it returns False.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return False

    # Adjust orientation if needed
    img = adjust_orientation(img)

    # Detect regions using YOLOv8
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # format: [x_min, y_min, x_max, y_max]

    if len(boxes) == 0:
        return False  # No region detected

    # Determine bounding box that encloses all detections
    x_min = int(np.min(boxes[:, 0]))
    y_min = int(np.min(boxes[:, 1]))
    x_max = int(np.max(boxes[:, 2]))
    y_max = int(np.max(boxes[:, 3]))

    # Crop the image to the region of interest (ROI)
    cropped_roi = img[y_min:y_max, x_min:x_max]

    # Create augmented versions: original, vertical flip, horizontal flip, and both flips
    augmented_images = {
        "original": cropped_roi,
        "flip_vertical": cv2.flip(cropped_roi, 0),
        "flip_horizontal": cv2.flip(cropped_roi, 1),
        "flip_both": cv2.flip(cropped_roi, -1)
    }

    predictions = []
    for aug_name, aug_img in augmented_images.items():
        processed = preprocess_for_model(aug_img)
        # Predict returns a probability; assuming model outputs a single value per sample
        pred = tf_model.predict(processed)[0][0]
        predictions.append(pred)

    # Average the prediction scores from the augmented images
    avg_score = np.mean(predictions)
    return True if avg_score >= THRESHOLD else False

if __name__ == "__main__":
    test_image = "image_name.jpg"  # Replace with your image file path
    result = aadhar(test_image)
    print("Aadhar detected:" if result else "Aadhar not detected.")
