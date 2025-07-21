import shutil
from pathlib import Path
import mediapipe as mp
import cupy
import torch
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw

import torchvision.transforms as transforms
from pipelines import DMVTONPipeline
from utils.torch_utils import select_device, get_ckpt, load_ckpt
from models.generators.mobile_unet import MobileNetV2_unet
from models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from Real_ESRGAN.inference_realesrgan_copy import run_realesrgan

def make_power_2(img, base=16, method=Image.BICUBIC):
    """Resize image so that both dimensions are multiples of base."""
    try:
        ow, oh = img.size  # PIL image
    except Exception:
        oh, ow = img.shape  # numpy image
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def transform_image(img, method=Image.BICUBIC, normalize=True):
    """
    Apply transformation pipeline similar to get_transform().
    Steps:
        - Resize to power of 2
        - Random horizontal flip (only in train mode)
        - Convert to tensor
        - Normalize to [-1, 1] if normalize=True
    """
    base = float(2**4)
    img = make_power_2(img, base, method)
    img = transforms.ToTensor()(img)
    if normalize:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img

def resize_with_aspect(image, target_size=(256, 192), pad_color=(255, 255, 255)):
    """
    Resize image to target_size while maintaining aspect ratio and padding.
    target_size: (height, width)
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]

    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with the scale
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded image
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def crop_person(image, mp_selfie_segmentation):
    h, w, _ = image.shape
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
        results = segmentor.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask > 0.5  # Boolean mask (True=person)

        # Create a white background
        white_bg = np.ones_like(image, dtype=np.uint8) * 255  # White image

        # Apply the mask: keep person where mask=True, else white
        segmented_person = np.where(mask[:, :, None], image, white_bg)
        
        # Find bounding box of the person
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return image  # No person found
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        box_w = x_max - x_min
        box_h = y_max - y_min

        # Only adjust if smaller than 192x256
        target_w, target_h = 192, 256
        if box_w < target_w or box_h < target_h:
            # Adjust width
            if box_w < target_w:
                pad_w = (target_w - box_w) // 2
                x_min = max(0, x_min - pad_w)
                x_max = min(w, x_min + target_w)

            # Adjust height
            if box_h < target_h:
                pad_h = (target_h - box_h) // 2
                y_min = max(0, y_min - pad_h)
                y_max = min(h, y_min + target_h)

            # Ensure final crop fits within image bounds
            x_min = max(0, min(x_min, w - target_w))
            x_max = min(w, x_min + target_w)
            y_min = max(0, min(y_min, h - target_h))
            y_max = min(h, y_min + target_h)

        cropped = segmented_person[y_min:y_max, x_min:x_max]
        return cropped

device = select_device(0)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
warp_model = AFWM(3, True).to(device)
gen_model = MobileNetV2_unet(7, 4).to(device)

warp_ckpt = get_ckpt("checkpoints/pf_warp_best.pt")
load_ckpt(warp_model, warp_ckpt)
warp_model.eval()

gen_ckpt = get_ckpt("checkpoints/pf_gen_best.pt")
load_ckpt(gen_model, gen_ckpt)
gen_model.eval()

upscaler = run_realesrgan(gpu_id=0)
print("models loaded")

def process_photo(img_dir, cloth_name, warp_model, gen_model, upscaler, device=select_device(0), pipeline=DMVTONPipeline()):
    with torch.no_grad():
        person_img = cv2.imread(img_dir)
        person_img = crop_person(person_img, mp_selfie_segmentation)
        person_img = resize_with_aspect(person_img, target_size=(256, 192))
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        person_img = Image.fromarray(person_img).convert('RGB')
        person_img = transform_image(person_img).unsqueeze(0).to(device)

        cloth_img = cv2.imread(f"cloth/{cloth_name}.jpg")
        cloth_img = resize_with_aspect(cloth_img, target_size=(256, 192))
        cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2RGB)
        cloth_img = Image.fromarray(cloth_img).convert('RGB')
        cloth_img = transform_image(cloth_img).unsqueeze(0).to(device)

        cloth_mask = cv2.imread(f"cloth_mask/{cloth_name}.jpg")
        cloth_mask = resize_with_aspect(cloth_mask, target_size=(256, 192))
        cloth_mask = cv2.cvtColor(cloth_mask, cv2.COLOR_BGR2RGB)
        cloth_mask = Image.fromarray(cloth_mask).convert('L')
        cloth_mask = transform_image(cloth_mask, method=Image.NEAREST, normalize=False).unsqueeze(0).to(device)
        print("data transformed")

        with cupy.cuda.Device(int(device.split(':')[-1])):
            p_tryon, warped_cloth = pipeline(warp_model, gen_model, person_img, cloth_img, cloth_mask, phase="test")
        img_tensor = p_tryon[0].squeeze()  # Take the first image in the batch

        # Convert to numpy and process
        cv_img = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2  # Normalize to [0, 1]
        cv_img = (cv_img * 255).astype(np.uint8)  # Scale to [0, 255]

        # Convert RGB to BGR for OpenCV
        # p_tryon = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        p_tryon = cv_img
        print("tryon generated")
        _, _, output = upscaler.enhance(p_tryon, has_aligned=False, only_center_face=False, paste_back=True)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        print("upscaled")
        img_save_name = os.path.splitext(os.path.basename(img_dir))[0]
        cv2.imwrite(f"results/{img_save_name}_out.jpg", output)
        print("saved")

def process_vid(vid_dir, cloth_name, warp_model, gen_model, upscaler, device=select_device(0), pipeline=DMVTONPipeline()):
    with torch.no_grad():
        cloth_img = cv2.imread(f"cloth/{cloth_name}.jpg")
        cloth_img = resize_with_aspect(cloth_img, target_size=(256, 192))
        cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2RGB)
        cloth_img = Image.fromarray(cloth_img).convert('RGB')
        cloth_img = transform_image(cloth_img).unsqueeze(0).to(device)

        cloth_mask = cv2.imread(f"cloth_mask/{cloth_name}.jpg")
        cloth_mask = resize_with_aspect(cloth_mask, target_size=(256, 192))
        cloth_mask = cv2.cvtColor(cloth_mask, cv2.COLOR_BGR2RGB)
        cloth_mask = Image.fromarray(cloth_mask).convert('L')
        cloth_mask = transform_image(cloth_mask, method=Image.NEAREST, normalize=False).unsqueeze(0).to(device)
        print("data transformed")

        cap = cv2.VideoCapture(vid_dir)
        if not cap.isOpened():
            print(f"Error: Cannot open video {vid_dir}")
            return

        # Video writer setup
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_save = os.path.splitext(os.path.basename(vid_dir))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"results/{vid_save[0]}_out{vid_save[1]}", fourcc, fps, (768, 1024))

        while True:
            ret, person_img = cap.read()
            if not ret:
                break

            person_img = crop_person(person_img, mp_selfie_segmentation)
            person_img = resize_with_aspect(person_img, target_size=(256, 192))
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_img = Image.fromarray(person_img).convert('RGB')
            person_img = transform_image(person_img).unsqueeze(0).to(device)

            with cupy.cuda.Device(int(device.split(':')[-1])):
                p_tryon, warped_cloth = pipeline(warp_model, gen_model, person_img, cloth_img, cloth_mask, phase="test")
            img_tensor = p_tryon[0].squeeze()  # Take the first image in the batch

            # Convert to numpy and process
            cv_img = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2  # Normalize to [0, 1]
            cv_img = (cv_img * 255).astype(np.uint8)  # Scale to [0, 255]

            # Convert RGB to BGR for OpenCV
            # p_tryon = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            p_tryon = cv_img
            print("tryon generated")
            _, _, output = upscaler.enhance(p_tryon, has_aligned=False, only_center_face=False, paste_back=True)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"results/1_out.jpg", output)

            out.write(output)

        cap.release()
        out.release()

import streamlit as st
import os
from PIL import Image
import base64
from io import BytesIO

# Streamlit app title (minimal main content)
import streamlit as st
import os
from PIL import Image
import base64
from io import BytesIO
import random

st.set_page_config(page_title="Virtual Try-On App", layout="wide")

# Tabs on top
tab1, tab2 = st.tabs(["👕 Virtual Try-On", "👗 Skin Tone-Based Recommendation"])

# =============== TAB 1: Virtual Try-On ==================
with tab1:
    st.title("Virtual Try-ON")
    sidebar_tab1, sidebar_tab2 = st.sidebar.tabs(["All Cloths", "Recommended Cloths"])
    cloth_folder = "cloth"

    # Initialize session state
    if 'loaded_cloths' not in st.session_state:
        st.session_state.loaded_cloths = []
    if 'load_index' not in st.session_state:
        st.session_state.load_index = 0
    if 'selected_cloth' not in st.session_state:
        st.session_state.selected_cloth = None
    if 'media_type' not in st.session_state:
        st.session_state.media_type = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'result_path' not in st.session_state:
        st.session_state.result_path = None

    # Load all cloth files (done once)
    if 'all_cloth_files' not in st.session_state:
        st.session_state.all_cloth_files = [f for f in os.listdir(cloth_folder) if f.endswith('.jpg')]
        random.shuffle(st.session_state.all_cloth_files)

    # Load initial batch
    if len(st.session_state.loaded_cloths) == 0:
        st.session_state.loaded_cloths = st.session_state.all_cloth_files[:30]
        st.session_state.load_index = 30

    # ----------- TAB 1: ALL CLOTHS -----------
    with sidebar_tab1:
        sidebar_tab1.title("Cloth Images")
        cols = sidebar_tab1.columns(3)
        for i, cloth in enumerate(st.session_state.loaded_cloths):
            col = cols[i % 3]
            img_path = os.path.join(cloth_folder, cloth)
            try:
                img = Image.open(img_path).resize((96, 128))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                selected_class = "selected" if st.session_state.selected_cloth == cloth else ""
                col.markdown(
                    f'<img class="{selected_class}" src="data:image/jpeg;base64,{img_base64}" style="width:100%;">',
                    unsafe_allow_html=True
                )
                if col.button(cloth, key=f"cloth_name_{cloth}_{i}"):
                    st.session_state.selected_cloth = cloth
                    st.rerun()
            except Exception as e:
                col.write(f"Error loading {cloth}: {e}")

        # Load More button
        if st.session_state.load_index < len(st.session_state.all_cloth_files):
            if sidebar_tab1.button("Load More"):
                next_batch = st.session_state.all_cloth_files[st.session_state.load_index:st.session_state.load_index + 30]
                st.session_state.loaded_cloths.extend(next_batch)
                st.session_state.load_index += 30
                st.rerun()

    # ----------- TAB 2: RECOMMENDED CLOTHS -----------
    with sidebar_tab2:
        sidebar_tab2.title("Recommended Cloths")

        try:
            import json
            with open("final_outfit.json", "r") as f:
                recommended_data = json.load(f)

            recommended_cloths = [os.path.basename(pair["topwear"]) for pair in recommended_data.get("pairs", [])]
        except Exception as e:
            recommended_cloths = []
            sidebar_tab2.error(f"Error loading recommended cloths: {e}")

        if recommended_cloths:
            cols = sidebar_tab2.columns(3)
            for i, cloth in enumerate(recommended_cloths):
                img_path = os.path.join(cloth_folder, cloth)
                try:
                    img = Image.open(img_path).resize((96, 128))
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    selected_class = "selected" if st.session_state.selected_cloth == cloth else ""
                    cols[i % 3].markdown(
                        f'<img class="{selected_class}" src="data:image/jpeg;base64,{img_base64}" style="width:100%;">',
                        unsafe_allow_html=True
                    )
                    if cols[i % 3].button(cloth, key=f"rec_cloth_name_{cloth}_{i}"):
                        st.session_state.selected_cloth = cloth
                        st.rerun()
                except Exception as e:
                    cols[i % 3].write(f"Error loading {cloth}: {e}")
        else:
            sidebar_tab2.write("No recommended cloths found.")


    # Media type buttons
# ROW 1: Photo and Video buttons
# ROW 1: Photo and Video buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                display: block;
                margin: 0 auto;

            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("Photo", key="photo_button"):
            st.session_state.media_type = "photo"
            st.session_state.uploaded_file = None
            st.session_state.result_path = None
            st.rerun()

    with col3:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        if st.button("Video", key="video_button"):
            st.session_state.media_type = "video"
            st.session_state.uploaded_file = None
            st.session_state.result_path = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # FILE UPLOADER (after selecting media type)
# FILE UPLOADER (after selecting media type)
    if st.session_state.media_type:
        if st.session_state.media_type == "photo":
            uploaded_file = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"], key="photo_uploader")
        else:
            uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"], key="video_uploader")

        if uploaded_file is not None:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state.uploaded_file = file_path  # <<< CRUCIAL
            st.success(f"{st.session_state.media_type.capitalize()} uploaded successfully!")

    # ROW 2: Try On button

    col4, col5, col6 = st.columns(3)
    with col5:
        try_on_disabled = not (st.session_state.uploaded_file and st.session_state.selected_cloth)
        print(f"DEBUG: Uploaded={st.session_state.uploaded_file}, Cloth={st.session_state.selected_cloth}")  # TEMP DEBUG
        if st.button("Try On", disabled=try_on_disabled, key="try_on_button_center"):
            cloth_name = st.session_state.selected_cloth.split(".")[0]
            media_path = st.session_state.uploaded_file
            if st.session_state.media_type == "photo":
                process_photo(media_path, cloth_name, warp_model, gen_model, upscaler, device, DMVTONPipeline())
                result_name = os.path.splitext(os.path.basename(media_path))[0] + "_out.jpg"
                st.session_state.result_path = f"results/{result_name}"
            else:
                process_vid(media_path, cloth_name, warp_model, gen_model, upscaler, device, DMVTONPipeline())
                result_name = os.path.splitext(os.path.basename(media_path))[0] + "_out" + os.path.splitext(media_path)[1]
                st.session_state.result_path = f"results/{result_name}"
            st.rerun()
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            position: fixed;
            right: 0;
            left: auto;
            top: var(--header-height, 3.5rem);  /* Adjust for header */
            height: calc(100vh - var(--header-height, 3.5rem));
            overflow-y: auto;  /* Primary scrollbar */
            overflow-x: hidden;  /* No horizontal scroll */
            width: 370px !important;  /* Adjust width as needed */
            z-index: 1;
        }
        section[data-testid="stSidebar"] > div {
            height: auto;  /* Allow natural height */
            overflow: hidden;  /* Disable inner scrolling */
        }
        main {
            margin-right: 350px !important;  /* Match sidebar width */
        }
        /* Hover glow effect on image */
        [data-testid="stSidebar"] img:hover {
            border: 2px solid red !important;
            box-shadow: 0 0 10px red !important;
            cursor: pointer;
        }
        /* Persistent glow for selected image */
        [data-testid="stSidebar"] img.selected {
            border: 2px solid red !important;
            box-shadow: 0 0 10px red !important;
        }
        /* Style clickable name to look like text (no button appearance) */
        [data-testid="stSidebar"] button {
            background: none;
            border: none;
            padding: 0;
            color: inherit;
            text-align: left;
            cursor: pointer;
            font-size: inherit;
        }
        [data-testid="stSidebar"] button:hover {
            text-decoration: underline;  /* Optional: underline on hover for link-like feel */
        }
        /* Target and hide any potential inner scroll containers */
        [data-testid="stSidebar"] > div > div {
            overflow: hidden !important;
        }
        /* Ensure columns don't introduce scrolls */
        [data-testid="column"] {
            overflow: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Display result (centered)
    if st.session_state.result_path:
        col_left, col_center, col_right = st.columns([1, 2, 1])  # Middle column is wider
        with col_center:
            if st.session_state.media_type == "photo":
                st.image(st.session_state.result_path, caption="Try-On Result",width=450)
            else:
                st.video(st.session_state.result_path)


# =============== TAB 2: Random Content ==================
# =============== TAB 2: Fashion Recommendation ==================
with tab2:
    st.title("👗 Skin Tone-Based Fashion Recommendation")

    import json
    import mediapipe as mp
    import cv2
    import numpy as np
    from PIL import Image

    # Load JSON data
    with open("fashion_recommend.json", "r") as f:
        fashion_data = json.load(f)

    # Skin tone classifier
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

    # Extract skin tone from face
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

    # UI Elements
    gender = st.radio("Select your gender:", ["male", "female"])
    uploaded_file = st.file_uploader("Upload a clear image of your face 👇", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        bgr = extract_skin_tone_from_image(image)
        if bgr:
            tone = classify_skin_tone(bgr)
            st.success(f"🎯 Detected Skin Tone: **{tone}**")
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

