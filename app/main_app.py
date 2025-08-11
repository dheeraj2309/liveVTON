# ==============================================================================
#                 AI FASHION SUITE - FINAL & COMPLETE SCRIPT
# ==============================================================================

# --- Section 1: Imports ---
import torch
import open_clip
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import json
import cv2
import numpy as np
import mediapipe as mp
import cupy
import shutil
import random
import torchvision.transforms as transforms
from flask import Flask, render_template, request, session, redirect, url_for, flash, send_from_directory, jsonify
from flask_session import Session
import gc

# --- Import Custom Modules ---
from llmprompt import generate_initial_outfit_prompts, refine_single_item_prompt
from pipelines import DMVTONPipeline
from utils.torch_utils import select_device, get_ckpt, load_ckpt
from models.generators.res_unet import ResUnetGenerator
from models.warp_modules.afwm import AFWM
from Real_ESRGAN.inference_realesrgan_copy import run_realesrgan

# ==============================================================================
# --- Section 2: App Initialization & Global Configuration ---
# ==============================================================================
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configure Server-Side Sessions ---
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Define Reliable Absolute Paths for All Assets ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TOPWEAR_DIR = os.path.join(APP_ROOT, "clothes_tryon_dataset", "train", "cloth")
BOTTOMWEAR_DIR = os.path.join(APP_ROOT, "pants_dataset", "Images_BW")
CLOTH_DIR = os.path.join(APP_ROOT, "cloth")
CLOTH_MASK_DIR = os.path.join(APP_ROOT, "cloth_mask")

# --- Ensure Required Directories Exist ---
os.makedirs(os.path.join(APP_ROOT, "static", "results"), exist_ok=True)
os.makedirs(os.path.join(APP_ROOT, "static", "uploaded"), exist_ok=True)
os.makedirs(CLOTH_DIR, exist_ok=True)
os.makedirs(CLOTH_MASK_DIR, exist_ok=True)

# ==============================================================================
# --- Section 3: Model & Data Loading (Once at Startup) ---
# ==============================================================================

def load_clip_model():
    print("--- Loading fine-tuned CLIP model (FashionSeek)... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = os.path.join(APP_ROOT, "my_final_checkpoints", "my_model_prompt_eng_epoch_5.pt")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded fine-tuned CLIP model.")
    else:
        print("Fine-tuned model not found. Using pre-trained LAION model.")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    model.eval()
    return model, tokenizer, preprocess, device

def index_dataset(dataset_name, image_root, csv_path, embeddings_cache_path, clip_model, clip_preprocess, device):
    embeddings_cache_path = os.path.join(APP_ROOT, embeddings_cache_path)
    csv_path = os.path.join(APP_ROOT, csv_path)
    
    if not os.path.exists(embeddings_cache_path):
        df = pd.read_csv(csv_path)
        image_features_list = []
        with torch.no_grad():
            for filename in tqdm(df["filename"].tolist(), desc=f"Indexing {dataset_name}"):
                image_path = os.path.join(image_root, filename)
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    image_input = clip_preprocess(image).unsqueeze(0).to(device)
                    features = clip_model.encode_image(image_input)
                    features /= features.norm(dim=-1, keepdim=True)
                    image_features_list.append(features)
        image_features = torch.cat(image_features_list)
        torch.save(image_features, embeddings_cache_path)
    
    image_features = torch.load(embeddings_cache_path, map_location=device)
    df = pd.read_csv(csv_path)
    image_paths = [os.path.join(image_root, f) for f in df["filename"].tolist()]
    return image_features, image_paths

def load_vto_models():
    print("--- Loading Virtual Try-On models... ---")
    device = select_device(0)
    warp_model = AFWM(3, True).to(device)
    gen_model = ResUnetGenerator(7, 4,5).to(device)
    
    warp_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_warp_last.pt")
    load_ckpt(warp_model, warp_ckpt)
    warp_model.eval()

    gen_ckpt = get_ckpt("checkpoints/checkpoints/PFAFN/pf_gen_last.pt")
    load_ckpt(gen_model, gen_ckpt)
    gen_model.eval()
    upscaler = run_realesrgan(gpu_id=0)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    print("VTO Models loaded successfully.")
    return warp_model, gen_model, upscaler, mp_selfie_segmentation, device

def load_all_cloth_files():
    print("--- Loading and shuffling all VTO cloth files... ---")
    all_files = [f for f in os.listdir(CLOTH_DIR) if f.endswith('.jpg')]
    random.shuffle(all_files)
    return all_files

# --- Execute Loading ---
clip_model, clip_tokenizer, clip_preprocess, clip_device = load_clip_model()
TOPWEAR_FEATURES, TOPWEAR_PATHS = index_dataset("Topwear", TOPWEAR_DIR, "train_captions_cleaned.csv", "topwear_database.pt", clip_model, clip_preprocess, clip_device)
BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS = index_dataset("Bottomwear", BOTTOMWEAR_DIR, "pants_captions_final.csv", "pants_database.pt", clip_model, clip_preprocess, clip_device)
warp_model, gen_model, upscaler, mp_selfie_segmentation, vto_device = load_vto_models()
ALL_CLOTH_FILES = load_all_cloth_files()


# ==============================================================================
# --- Section 4: Helper Functions ---
# ==============================================================================

def load_ai_stylist_models():
    """Checks if AI Stylist models were freed, and if so, reloads them."""
    global clip_model, clip_tokenizer, clip_preprocess, clip_device
    global TOPWEAR_FEATURES, TOPWEAR_PATHS, BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS
    
    if clip_model is None:
        print("--- Reloading AI Stylist models on-demand... ---")
        clip_model, clip_tokenizer, clip_preprocess, clip_device = load_clip_model()
        TOPWEAR_FEATURES, TOPWEAR_PATHS = index_dataset("Topwear", TOPWEAR_DIR, "train_captions_cleaned.csv", "topwear_database.pt", clip_model, clip_preprocess, clip_device)
        BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS = index_dataset("Bottomwear", BOTTOMWEAR_DIR, "pants_captions_final.csv", "pants_database.pt", clip_model, clip_preprocess, clip_device)
        print("--- AI Stylist models reloaded. ---")

# In main_app.py

# In main_app.py

def free_ai_stylist_models():
    """
    Frees memory used by the AI Stylist models and prints a detailed
    GPU memory report before and after for debugging.
    """
    global clip_model, clip_tokenizer, clip_preprocess, clip_device
    global TOPWEAR_FEATURES, TOPWEAR_PATHS, BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS

    if clip_model is not None:
        print("--- Freeing AI Stylist model memory for intensive task... ---")
        
        # --- DEBUG: Print memory summary BEFORE freeing ---
        print("\n=================== VRAM USAGE: BEFORE FREEING ===================")
        print(torch.cuda.memory_summary())
        print("====================================================================\n")

        try:
            # 1. Explicitly move large tensors from GPU to CPU to de-allocate VRAM
            clip_model.to('cpu')
            if TOPWEAR_FEATURES is not None:
                TOPWEAR_FEATURES.to('cpu')
            if BOTTOMWEAR_FEATURES is not None:
                BOTTOMWEAR_FEATURES.to('cpu')
        except Exception as e:
            print(f"Could not move all tensors to CPU: {e}")

        # 2. Delete the large Python objects to remove all references
        del clip_model
        del TOPWEAR_FEATURES
        del BOTTOMWEAR_FEATURES
        
        # 3. Set global variables to None so the app knows they are unloaded
        clip_model = None
        clip_tokenizer, clip_preprocess, clip_device = None, None, None
        TOPWEAR_FEATURES = None
        BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS = None, None

        # 4. Run garbage collection and clear the CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

        # --- DEBUG: Print memory summary AFTER freeing ---
        print("\n==================== VRAM USAGE: AFTER FREEING ====================")
        print(torch.cuda.memory_summary())
        print("===================================================================\n")
        
        print("--- Memory freeing process complete. ---")

def find_items_by_text(prompt, item_type, top_n=1):
    features, paths = (TOPWEAR_FEATURES, TOPWEAR_PATHS) if item_type == 'topwear' else (BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS)
    with torch.no_grad():
        text_features = clip_model.encode_text(clip_tokenizer([prompt]).to(clip_device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ features.T).squeeze(0)
        _, indices = torch.topk(similarity, k=top_n)
        return [paths[i] for i in indices]

def find_similar_items_by_image(image_path, item_type, top_n=15):
    features, paths = (TOPWEAR_FEATURES, TOPWEAR_PATHS) if item_type == 'topwear' else (BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS)
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0).to(clip_device)
        query_features = clip_model.encode_image(image_input)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        similarity = (query_features @ features.T).squeeze(0)
        _, indices = torch.topk(similarity, k=top_n + 1)
        return [p for p in [paths[i] for i in indices] if p != image_path][:top_n]

def initialize_session():
    if 'user_profile' not in session:
        session['user_profile'] = {"age": 25, "gender": "female", "occasion": "daily wear", "color": "neutral colors", "style_type": "casual", "comments": "I prefer comfortable clothes."}
    if 'topwear' not in session: session['topwear'] = {"status": "pending"}
    if 'bottomwear' not in session: session['bottomwear'] = {"status": "pending"}
    if 'final_outfit' not in session: session['final_outfit'] = {}

def save_final_outfit():
    filename = "final_outfit.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try: data = json.load(f)
            except json.JSONDecodeError: data = {"pairs": []}
    else:
        data = {"pairs": []}
    
    # Reconstruct full relative paths for saving
    topwear_filename = session.get("final_outfit", {}).get("topwear", "")
    bottomwear_filename = session.get("final_outfit", {}).get("bottomwear", "")
    
    new_pair = {
        "topwear": os.path.join("clothes_tryon_dataset/train/cloth", topwear_filename).replace("\\", "/"),
        "bottomwear": os.path.join("pants_dataset/Images_BW", bottomwear_filename).replace("\\", "/")
    }

    if new_pair["topwear"] and new_pair["bottomwear"] and new_pair not in data["pairs"]:
        data["pairs"].append(new_pair)
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return "Your final outfit has been added to `final_outfit.json`!"

# --- VTO & Skin Tone Helpers ---
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

def transform_image_vto(img, method=Image.BICUBIC, normalize=True):
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

def process_photo(img_dir, cloth_name, pipeline=DMVTONPipeline()):
    """
    Performs VTO on a single photo with integrated cropping and resizing.
    """
    with torch.no_grad():
        # --- Person Image Preprocessing ---
        person_img_cv2 = cv2.imread(img_dir)
        # Apply crop and resize logic
        person_img_cv2 = crop_person(person_img_cv2, mp_selfie_segmentation)
        person_img_cv2 = resize_with_aspect(person_img_cv2, target_size=(256, 192))
        # Convert to PIL and then to Tensor
        person_img_pil = Image.fromarray(cv2.cvtColor(person_img_cv2, cv2.COLOR_BGR2RGB)).convert('RGB')
        person_img = transform_image_vto(person_img_pil).unsqueeze(0).to(vto_device)

        # --- Cloth Image Preprocessing ---
        cloth_img_cv2 = cv2.imread(os.path.join(CLOTH_DIR, f"{cloth_name}.jpg"))
        # Apply resize logic
        cloth_img_cv2 = resize_with_aspect(cloth_img_cv2, target_size=(256, 192))
        # Convert to PIL and then to Tensor
        cloth_img_pil = Image.fromarray(cv2.cvtColor(cloth_img_cv2, cv2.COLOR_BGR2RGB)).convert('RGB')
        cloth_img = transform_image_vto(cloth_img_pil).unsqueeze(0).to(vto_device)

        # --- Cloth Mask Preprocessing ---
        cloth_mask_cv2 = cv2.imread(os.path.join(CLOTH_MASK_DIR, f"{cloth_name}.jpg"))
        # Apply resize logic
        cloth_mask_cv2 = resize_with_aspect(cloth_mask_cv2, target_size=(256, 192))
        # Convert to PIL (grayscale) and then to Tensor
        cloth_mask_pil = Image.fromarray(cv2.cvtColor(cloth_mask_cv2, cv2.COLOR_BGR2RGB)).convert('L')
        cloth_mask = transform_image_vto(cloth_mask_pil, method=Image.NEAREST, normalize=False).unsqueeze(0).to(vto_device)
        
        # --- Inference and Post-processing (Unchanged) ---
        with cupy.cuda.Device(int(vto_device.split(':')[-1])):
            p_tryon, _ = pipeline(warp_model, gen_model, person_img, cloth_img, cloth_mask, phase="test")
        
        cv_img = (p_tryon[0].squeeze().detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
        cv_img = (cv_img * 255).astype(np.uint8)
        _, _, output = upscaler.enhance(cv_img, has_aligned=False, only_center_face=False, paste_back=True)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        img_save_name = f"{os.path.splitext(os.path.basename(img_dir))[0]}_out.jpg"
        save_path = os.path.join(APP_ROOT, "static", "results", img_save_name)
        cv2.imwrite(save_path, output)
        return os.path.join("results", img_save_name).replace("\\", "/")

def process_video(vid_dir, cloth_name, pipeline=DMVTONPipeline()):
    """
    Performs VTO on a video with integrated cropping and resizing for each frame.
    """
    with torch.no_grad():
        # --- Cloth & Mask Preprocessing (Done once) ---
        cloth_img_cv2 = cv2.imread(os.path.join(CLOTH_DIR, f"{cloth_name}.jpg"))
        cloth_img_cv2 = resize_with_aspect(cloth_img_cv2, target_size=(256, 192))
        cloth_img_pil = Image.fromarray(cv2.cvtColor(cloth_img_cv2, cv2.COLOR_BGR2RGB)).convert('RGB')
        cloth_img = transform_image_vto(cloth_img_pil).unsqueeze(0).to(vto_device)

        cloth_mask_cv2 = cv2.imread(os.path.join(CLOTH_MASK_DIR, f"{cloth_name}.jpg"))
        cloth_mask_cv2 = resize_with_aspect(cloth_mask_cv2, target_size=(256, 192))
        cloth_mask_pil = Image.fromarray(cv2.cvtColor(cloth_mask_cv2, cv2.COLOR_BGR2RGB)).convert('L')
        cloth_mask = transform_image_vto(cloth_mask_pil, method=Image.NEAREST, normalize=False).unsqueeze(0).to(vto_device)

        # --- Video I/O Setup (Unchanged) ---
        cap = cv2.VideoCapture(vid_dir)
        if not cap.isOpened():
            print(f"Error: Cannot open video {vid_dir}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_extension = os.path.splitext(os.path.basename(vid_dir))[1]
        vid_save_name = f"{os.path.splitext(os.path.basename(vid_dir))[0]}_out{vid_extension}"
        output_width, output_height = 192 * 4, 256 * 4
        save_path = os.path.join(APP_ROOT, "static", "results", vid_save_name)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(save_path, fourcc, fps, (output_width, output_height))

        frame_count = 0
        while True:
            ret, person_frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_count}...")
            frame_count += 1

            # --- Per-Frame Preprocessing ---
            person_frame_processed = crop_person(person_frame, mp_selfie_segmentation)
            person_frame_processed = resize_with_aspect(person_frame_processed, target_size=(256, 192))
            person_img_pil = Image.fromarray(cv2.cvtColor(person_frame_processed, cv2.COLOR_BGR2RGB)).convert('RGB')
            person_img = transform_image_vto(person_img_pil).unsqueeze(0).to(vto_device)

            # --- Inference and Post-processing (Unchanged) ---
            with cupy.cuda.Device(int(vto_device.split(':')[-1])):
                p_tryon, _ = pipeline(warp_model, gen_model, person_img, cloth_img, cloth_mask, phase="test")

            cv_img = (p_tryon[0].squeeze().detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
            cv_img = (cv_img * 255).astype(np.uint8)
            _, _, output = upscaler.enhance(cv_img, has_aligned=False, only_center_face=False, paste_back=True)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            if output.shape[1] != output_width or output.shape[0] != output_height:
                output = cv2.resize(output, (output_width, output_height))
            out.write(output)

        cap.release()
        out.release()
        print("Video processing complete.")
        return os.path.join("results", vid_save_name).replace("\\", "/")

def classify_skin_tone(bgr):
    b, g, r = bgr
    brightness = (r + g + b) / 3
    if brightness > 180: return "Fair"
    elif brightness > 150: return "Light"
    elif brightness > 120: return "Medium"
    elif brightness > 90: return "Olive"
    else: return "Dark"

def extract_skin_tone_from_image(image_path):
    image_np = np.array(Image.open(image_path).convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_np)
        if not results.detections: return None
        bbox = results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = image_np.shape
        x1, y1 = int(bbox.xmin * iw), int(bbox.ymin * ih)
        w, h = int(bbox.width * iw), int(bbox.height * ih)
        face = image_bgr[y1:y1+h, x1:x1+w]
        face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
        skin_pixels = cv2.bitwise_and(face, face, mask=mask)[np.where(mask != 0)]
        if len(skin_pixels) == 0: return None
        return tuple(int(c) for c in np.mean(skin_pixels, axis=0))


# ==============================================================================
# --- Section 5: Flask Routes ---
# ==============================================================================

@app.route('/images/<filename>')
def serve_image(filename):
    if os.path.exists(os.path.join(TOPWEAR_DIR, filename)):
        return send_from_directory(TOPWEAR_DIR, filename)
    elif os.path.exists(os.path.join(BOTTOMWEAR_DIR, filename)):
        return send_from_directory(BOTTOMWEAR_DIR, filename)
    else:
        return 'Image not found', 404

@app.route('/cloth/<filename>')
def serve_cloth_image(filename):
    return send_from_directory(CLOTH_DIR, filename)

@app.route('/')
def ai_stylist_page():
    load_ai_stylist_models()
    initialize_session()
    return render_template('ai_stylist.html')

@app.route('/update', methods=['POST'])
def update():
    load_ai_stylist_models()
    initialize_session()
    action = request.form.get('action')
    toast_message = None
    template_context = {}

    if action == 'generate':
        session['user_profile'] = { "age": int(request.form.get('age')), "gender": request.form.get('gender'), "occasion": request.form.get('occasion'), "color": request.form.get('color'), "style_type": request.form.get('style_type'), "comments": request.form.get('comments')}
        prompts = generate_initial_outfit_prompts(session['user_profile'])
        if prompts:
            top_full_path = find_items_by_text(prompts['topwear'], 'topwear')[0]
            bottom_full_path = find_items_by_text(prompts['bottomwear'], 'bottomwear')[0]
            session['topwear'] = {"status": "initial", "prompt": prompts['topwear'], "path": os.path.basename(top_full_path)}
            session['bottomwear'] = {"status": "initial", "prompt": prompts['bottomwear'], "path": os.path.basename(bottom_full_path)}
        else:
            toast_message = {"type": "error", "text": "Failed to generate prompts."}

    elif action == 'interact':
        item_type = request.form.get('item_type')
        interaction = request.form.get('interaction')
        if interaction == 'like':
            session[item_type]['status'] = 'liked'
            liked_item_filename = session[item_type]['path']
            base_dir = TOPWEAR_DIR if item_type == 'topwear' else BOTTOMWEAR_DIR
            full_item_path = os.path.join(base_dir, liked_item_filename)
            similar_paths = find_similar_items_by_image(full_item_path, item_type)
            template_context['similar_items_data'] = {
                "type": item_type,
                "paths": [os.path.basename(p) for p in similar_paths]
            }
        elif interaction == 'dislike':
            session[item_type]['status'] = 'feedback'

    elif action == 'refine':
        item_type = request.form.get('item_type')
        feedback = request.form.get('feedback')
        new_prompt = refine_single_item_prompt(item_type, session[item_type]['prompt'], feedback, session['user_profile'])
        new_path = os.path.basename(find_items_by_text(new_prompt, item_type)[0])
        session[item_type] = {"status": "initial", "prompt": new_prompt, "path": new_path}

    elif action == 'choose':
        item_type = request.form.get('item_type')
        chosen_path = request.form.get('chosen_path')
        session[item_type]['status'] = 'chosen'
        session[item_type]['path'] = chosen_path
        session['final_outfit'][item_type] = chosen_path

    elif action == 'save':
        message = save_final_outfit()
        toast_message = {"type": "success", "text": message}

    elif action == 'reset':
        session.clear()
        initialize_session()

    session.modified = True
    rendered_html = render_template('_dynamic_content.html', **template_context)
    return jsonify(html=rendered_html, toast=toast_message)

@app.route('/virtual_tryon', methods=['GET', 'POST'])
def virtual_tryon_page():
    result_path = None
    result_type = None  # To tell the template if it's a photo or video
    if request.method == 'POST':
        user_file = request.files.get('user_file')
        cloth_name = request.form.get('cloth_name')
        media_type = request.form.get('media_type') # <-- Get the media type

        if user_file and cloth_name and media_type and user_file.filename != '':
            free_ai_stylist_models()
            upload_path = os.path.join(APP_ROOT, "static", "uploaded", user_file.filename)
            user_file.save(upload_path)
            try:
                # --- Call the correct function based on media type ---
                if media_type == 'photo':
                    result_path = process_photo(upload_path, cloth_name.replace('.jpg', ''))
                    result_type = 'photo'
                elif media_type == 'video':
                    result_path = process_video(upload_path, cloth_name.replace('.jpg', ''))
                    result_type = 'video'

                flash('Try-on generated successfully!', 'success')
            except Exception as e:
                print(f"An error occurred: {e}")
                flash(f'An error occurred during processing: {e}', 'danger')
        else:
            flash('Please select media type, upload a file, and choose a cloth.', 'warning')

    try:
        with open("final_outfit.json", "r") as f:
            pairs = json.load(f).get("pairs", [])
            # The reversed() function processes the list from last to first
            reco_files = [os.path.basename(p["topwear"]) for p in reversed(pairs)]
    except (FileNotFoundError, json.JSONDecodeError):
        reco_files = []

    return render_template(
        'virtual_tryon.html',
        initial_all_cloths=ALL_CLOTH_FILES[:30],
        initial_reco_cloths=reco_files[:30],
        all_cloths_total=len(ALL_CLOTH_FILES),
        reco_cloths_total=len(reco_files),
        result_path=result_path,
        result_type=result_type # <-- Pass the result type to the template
    )
@app.route('/load_cloths')
def load_cloths():
    cloth_type = request.args.get('type', 'all')
    offset = request.args.get('offset', 0, type=int)
    limit = 30
    if cloth_type == 'all':
        cloth_list = ALL_CLOTH_FILES
    else: # recommended
        try:
            with open("final_outfit.json", "r") as f:
                cloth_list = [os.path.basename(p["topwear"]) for p in json.load(f).get("pairs", [])]
        except (FileNotFoundError, json.JSONDecodeError):
            cloth_list = []
    next_batch = cloth_list[offset:offset + limit]
    return jsonify(cloths=next_batch)

@app.route('/skin_tone_reco', methods=['GET', 'POST'])
def skin_tone_reco_page():
    results = None
    if request.method == 'POST':
        user_face_file = request.files.get('user_face_file')
        gender = request.form.get('gender')

        if user_face_file and gender and user_face_file.filename != '':
            upload_path = os.path.join(APP_ROOT, "static", "uploaded", user_face_file.filename)
            user_face_file.save(upload_path)
            
            try:
                bgr = extract_skin_tone_from_image(upload_path)
                if bgr:
                    tone = classify_skin_tone(bgr)
                    with open("fashion_recommend.json", "r") as f:
                        fashion_data = json.load(f)
                    recommendations = fashion_data.get(gender, {}).get(tone, {}).get("combinations", [])
                    results = {"tone": tone, "recommendations": recommendations}
                else:
                    flash('Could not detect face or skin tone. Please try a clearer picture.', 'warning')
            except Exception as e:
                flash(f'An error occurred: {e}', 'danger')
        else:
            flash('Please upload a face image and select a gender.', 'warning')

    return render_template('skin_tone_reco.html', results=results)

# ==============================================================================
# --- Section 6: App Execution ---
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)