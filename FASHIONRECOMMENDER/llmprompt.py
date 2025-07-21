# ==============================================================================
#           ADVANCED OUTFIT PROMPT GENERATOR (llmprompt.py) - FIXED VERSION
# ==============================================================================
# This module handles all interactions with the LLM (Gemini). It can generate
# initial outfits from a rich user profile and refine suggestions based on
# feedback and a history of disliked items.
# ==============================================================================
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

try:
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
except (TypeError, ValueError) as e:
    print(f"ERROR: Google API Key not configured correctly. Details: {e}")
    model = None

USER_PREFERENCES_FILE = "user_dislike_history.json"

# ==============================================================================
#                          USER PREFERENCE UTILITIES
# ==============================================================================
def load_user_preferences():
    """Loads the user's dislike history from a JSON file."""
    if not os.path.exists(USER_PREFERENCES_FILE):
        return []
    with open(USER_PREFERENCES_FILE, 'r') as f:
        return json.load(f)

def save_user_preferences(dislike_history):
    """Saves the user's dislike history to a JSON file."""
    with open(USER_PREFERENCES_FILE, 'w') as f:
        json.dump(dislike_history, f, indent=2)

# ==============================================================================
#                       INITIAL PROMPT GENERATION FUNCTION
# ==============================================================================
def generate_initial_outfit_prompts(profile: dict) -> dict:
    """
    Generates complementing prompts for a top and bottom based on a rich user profile.
    Ensures output is valid JSON with string values for "topwear" and "bottomwear".
    """
    if model is None:
        return {"error": "LLM model not configured."}

    dislike_history = load_user_preferences()
    history_prompt = ""
    if dislike_history:
        history_str = "\n- ".join(dislike_history)
        history_prompt = (
            "CRITICAL: Avoid suggestions with these characteristics based on user's past dislikes:\n"
            f"- {history_str}"
        )

    context_prompt = f"""
    You are an expert fashion stylist. A user's profile is:
    - Age: {profile.get('age')}
    - Gender: {profile.get('gender')}
    - Occasion: {profile.get('occasion')}
    - Color Preference: {profile.get('color')}
    - Style Type: {profile.get('style_type')}
    - Additional Comments: {profile.get('comments')}

    {history_prompt}

    Based on this profile, suggest a complete, stylish, and complementary outfit.
    Provide a concise, under-20-word, text-to-image prompt for each item.
    Your output MUST be a valid JSON with "topwear" and "bottomwear" keys. No other text.
    """

    try:
        response = model.generate_content(context_prompt)
        raw_output = response.text.strip().replace("```json", "").replace("```", "")
        print("🔍 Raw Gemini Output:", raw_output)  # Debugging aid

        parsed = json.loads(raw_output)

        # --- Clean nested prompt fields if needed ---
        if isinstance(parsed.get("topwear"), dict):
            parsed["topwear"] = parsed["topwear"].get("prompt", "")
        if isinstance(parsed.get("bottomwear"), dict):
            parsed["bottomwear"] = parsed["bottomwear"].get("prompt", "")

        # Ensure both keys exist
        parsed.setdefault("topwear", "")
        parsed.setdefault("bottomwear", "")

        return parsed

    except Exception as e:
        print(f" Error parsing LLM output: {e}")
        return {"error": "Failed to parse Gemini response."}

# ==============================================================================
#                   SINGLE ITEM PROMPT REFINEMENT FUNCTION
# ==============================================================================
def refine_single_item_prompt(item_type: str, original_prompt: str, feedback: str, profile: dict) -> str:
    """
    Generates a new prompt for a single item based on user feedback.
    Also updates user's dislike history to avoid similar suggestions.
    """
    if model is None:
        return "Error: LLM model not configured."

    # --- Update dislike history ---
    dislike_history = load_user_preferences()
    dislike_reason = f"Disliked '{original_prompt}' because: '{feedback}'"
    if dislike_reason not in dislike_history:
        dislike_history.append(dislike_reason)
        save_user_preferences(dislike_history)

    history_str = "\n- ".join(dislike_history)

    refinement_context = f"""
    A user's profile is: {json.dumps(profile)}.
    The user was shown a {item_type} based on the description: "{original_prompt}".
    Their new feedback is: "{feedback}".

    Here is the user's full dislike history for context:
    - {history_str}

    Based on all this information, generate a new, improved, under-20-word text-to-image prompt
    for the {item_type} that specifically addresses the new feedback while respecting past dislikes.
    Return only the prompt text and nothing else.
    """

    try:
        response = model.generate_content(refinement_context)
        new_prompt = response.text.strip()
        print(f"🛠 Refined {item_type} prompt: {new_prompt}")
        return new_prompt
    except Exception as e:
        print(f"Error refining prompt: {e}")
        return "Error generating refined prompt."
