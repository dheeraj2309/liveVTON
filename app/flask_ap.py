# ==============================================================================
#           INTERACTIVE AI STYLIST (flask_app.py) - JS ENHANCED
# ==============================================================================
import torch
import open_clip
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import json
from flask import Flask, render_template, request, session, redirect, url_for, flash, send_from_directory, jsonify

# --- Import your custom AI modules ---
from llmprompt import generate_initial_outfit_prompts, refine_single_item_prompt

# ==============================================================================
# --- 1. APP & MODEL CONFIGURATION ---
# ==============================================================================
app = Flask(__name__)
app.secret_key = os.urandom(24)

# (Model and Data Loading code remains exactly the same as before...)
# --- Model Loading (Runs once at startup) ---
def load_clip_model():
    """Loads the fine-tuned FashionSeek model just once."""
    print("--- Loading fine-tuned CLIP model (FashionSeek)... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "my_final_checkpoints/my_model_prompt_eng_epoch_5.pt"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded fine-tuned CLIP model.")
    else:
        print("Fine-tuned model not found. Loading pre-trained LAION model as a fallback.")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    model.eval()
    return model, tokenizer, preprocess, device

def index_dataset(dataset_name, image_root, csv_path, embeddings_cache_path, clip_model, clip_preprocess, device):
    """A reusable function to load or create indexed image features for any dataset."""
    if os.path.exists(embeddings_cache_path):
        image_features = torch.load(embeddings_cache_path, map_location=device)
    else:
        df = pd.read_csv(csv_path); image_filenames = df["filename"].tolist()
        image_features_list = []
        with torch.no_grad():
            for filename in tqdm(image_filenames, desc=f"Indexing {dataset_name}"):
                image_path = os.path.join(image_root, filename)
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB"); image_input = clip_preprocess(image).unsqueeze(0).to(device)
                    features = clip_model.encode_image(image_input); features /= features.norm(dim=-1, keepdim=True)
                    image_features_list.append(features)
        image_features = torch.cat(image_features_list); torch.save(image_features, embeddings_cache_path)
    df = pd.read_csv(csv_path)
    image_paths = [os.path.abspath(os.path.join(image_root, f)) for f in df["filename"].tolist()]
    return image_features, image_paths

clip_model, clip_tokenizer, clip_preprocess, device = load_clip_model()
TOPWEAR_FEATURES, TOPWEAR_PATHS = index_dataset("Topwear", "clothes_tryon_dataset/train/cloth", "train_captions_cleaned.csv", "topwear_database.pt", clip_model, clip_preprocess, device)
BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS = index_dataset("Bottomwear", "pants_dataset/Images_BW", "pants_captions_final.csv", "pants_database.pt", clip_model, clip_preprocess, device)


# ==============================================================================
# --- 2. SEARCH & HELPER FUNCTIONS ---
# ==============================================================================
# (find_items_by_text and find_similar_items_by_image functions are the same)
def find_items_by_text(prompt, item_type, top_n=1):
    features, paths = (TOPWEAR_FEATURES, TOPWEAR_PATHS) if item_type == 'topwear' else (BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS)
    with torch.no_grad():
        text_features = clip_model.encode_text(clip_tokenizer([prompt]).to(device)); text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ features.T).squeeze(0); _, indices = torch.topk(similarity, k=top_n)
        return [paths[i] for i in indices]

def find_similar_items_by_image(image_path, item_type, top_n=15):
    features, paths = (TOPWEAR_FEATURES, TOPWEAR_PATHS) if item_type == 'topwear' else (BOTTOMWEAR_FEATURES, BOTTOMWEAR_PATHS)
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB"); image_input = clip_preprocess(image).unsqueeze(0).to(device)
        query_features = clip_model.encode_image(image_input); query_features /= query_features.norm(dim=-1, keepdim=True)
        similarity = (query_features @ features.T).squeeze(0); _, indices = torch.topk(similarity, k=top_n + 1)
        return [p for p in [paths[i] for i in indices] if p != image_path][:top_n]

def save_final_outfit():
    filename = "final_outfit.json"

    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"pairs": []}
    else:
        data = {"pairs": []}

    # Add the current outfit pair
    new_pair = {
        "topwear": session.get("final_outfit", {}).get("topwear", ""),
        "bottomwear": session.get("final_outfit", {}).get("bottomwear", "")
    }

    # Avoid adding empty or duplicate entries
    if new_pair["topwear"] and new_pair["bottomwear"] and new_pair not in data["pairs"]:
        data["pairs"].append(new_pair)

    # Save updated data
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return "Your final outfit has been added to `final_outfit.json`!"



# ==============================================================================
# --- 3. FLASK ROUTES & UI ---
# ==============================================================================

def initialize_session():
    """Helper to set default session state."""
    if 'user_profile' not in session:
        session['user_profile'] = {"age": 25, "gender": "female", "occasion": "daily wear", "color": "neutral colors", "style_type": "casual", "comments": "I prefer comfortable clothes."}
    if 'topwear' not in session:
        session['topwear'] = {"status": "pending"}
    if 'bottomwear' not in session:
        session['bottomwear'] = {"status": "pending"}
    if 'similar_items' not in session:
        session['similar_items'] = {"type": None, "paths": []}
    if 'final_outfit' not in session:
        session['final_outfit'] = {}

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.dirname(filename), os.path.basename(filename))

@app.route('/')
def index():
    initialize_session()
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    """A single endpoint to handle all AJAX interactions."""
    initialize_session()
    action = request.form.get('action')
    toast_message = None

    if action == 'generate':
        session['user_profile'] = { "age": int(request.form.get('age')), "gender": request.form.get('gender'), "occasion": request.form.get('occasion'), "color": request.form.get('color'), "style_type": request.form.get('style_type'), "comments": request.form.get('comments')}
        prompts = generate_initial_outfit_prompts(session['user_profile'])
        if prompts:
            top_path = find_items_by_text(prompts['topwear'], 'topwear')[0]
            bottom_path = find_items_by_text(prompts['bottomwear'], 'bottomwear')[0]
            session['topwear'] = {"status": "initial", "prompt": prompts['topwear'], "path": top_path}
            session['bottomwear'] = {"status": "initial", "prompt": prompts['bottomwear'], "path": bottom_path}
            session['similar_items'] = {"type": None, "paths": []}; session['final_outfit'] = {}
        else:
            toast_message = {"type": "error", "text": "Failed to generate prompts."}

    elif action == 'interact':
        item_type = request.form.get('item_type')
        interaction = request.form.get('interaction')
        if interaction == 'like':
            session[item_type]['status'] = 'liked'
            session['similar_items'] = {"type": item_type, "paths": find_similar_items_by_image(session[item_type]['path'], item_type)}
        elif interaction == 'dislike':
            session[item_type]['status'] = 'feedback'

    elif action == 'refine':
        item_type = request.form.get('item_type')
        feedback = request.form.get('feedback')
        original_prompt = session[item_type]['prompt']
        new_prompt = refine_single_item_prompt(item_type, original_prompt, feedback, session['user_profile'])
        new_path = find_items_by_text(new_prompt, item_type)[0]
        session[item_type] = {"status": "initial", "prompt": new_prompt, "path": new_path}

    elif action == 'choose':
        item_type = request.form.get('item_type')
        chosen_path = request.form.get('chosen_path')
        session[item_type]['status'] = 'chosen'
        session[item_type]['path'] = chosen_path
        session['final_outfit'][item_type] = chosen_path
        session['similar_items'] = {"type": None, "paths": []}

    elif action == 'save':
        message = save_final_outfit()
        toast_message = {"type": "success", "text": message}

    elif action == 'reset':
        session.clear()
        initialize_session()

    session.modified = True
    # Render the dynamic part of the page and return as JSON
    rendered_html = render_template('_dynamic_content.html')
    return jsonify(html=rendered_html, toast=toast_message)

if __name__ == '__main__':
    app.run(debug=True)
