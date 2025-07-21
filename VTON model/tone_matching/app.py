import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
from PIL import Image

# Load JSON recommendation data
with open("fashion_recommend.json", "r") as f:
    fashion_data = json.load(f)

# Classify skin tone from BGR values
def classify_skin_tone(bgr):
    b, g, r = bgr
    brightness = (r + g + b) / 3
    if brightness > 180:
        return "Fair"
    elif brightness > 150:
        return "Light"
    elif brightness > 120:
        return "Medium"
    elif brightness > 90:
        return "Olive"
    else:
        return "Dark"

# Extract average skin color from face using MediaPipe
def extract_skin_tone_from_image(image):
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_np)
        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = image_np.shape
        x1, y1 = int(bbox.xmin * iw), int(bbox.ymin * ih)
        w, h = int(bbox.width * iw), int(bbox.height * ih)
        face = image_bgr[y1:y1+h, x1:x1+w]

        face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
        skin = cv2.bitwise_and(face, face, mask=mask)

        skin_pixels = skin[np.where(mask != 0)]
        if len(skin_pixels) == 0:
            return None

        avg_color = np.mean(skin_pixels, axis=0)
        return tuple(int(c) for c in avg_color)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Fashion Tone Recommender", layout="centered")
st.title("👗 Skin Tone-Based Fashion Recommendation")

# Gender selection
gender = st.radio("Select your gender:", ["male", "female"])

# Image upload
uploaded_file = st.file_uploader("Upload a clear image of your face 👇", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    bgr = extract_skin_tone_from_image(image)
    if bgr:
        tone = classify_skin_tone(bgr)
        st.success(f"🎯 Detected Skin Tone: **{tone}**")

        # Fetch fashion combinations from JSON
        try:
            combos = fashion_data[gender][tone]["combinations"]

            st.subheader("🧥 Outfit Recommendations:")
            for i, combo in enumerate(combos, 1):
                st.markdown(
                    f"**Combo {i}:** 🟦 *{combo['upperwear']}* upperwear + 👖 *{combo['bottomwear']}* bottomwear"
                )
        except KeyError:
            st.warning("No recommendations found for this gender and skin tone.")
    else:
        st.error("❌ Could not detect face or skin tone. Try a clearer photo.")
