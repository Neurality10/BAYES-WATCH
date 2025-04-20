# --- START OF vision_descriptor.py (Using the "Old" Structure) ---

import os
import tempfile
import requests
import cloudinary
import cloudinary.uploader
from openai import OpenAI
from PIL import Image

# ==== CONFIGURATION ====
OPENROUTER_API_KEY = "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" # <-- Make sure this is your actual key
CLOUDINARY_CLOUD_NAME = "dddnjdjfb" # <-- Your Cloudinary cloud name
CLOUDINARY_API_KEY = "628645996396588"
CLOUDINARY_API_SECRET = "w5AmZBd7GEjf8_-k9A7ocXHv3zY"
API_BASE = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen2.5-vl-32b-instruct:free"
API_TIMEOUT = 180

# ==== Cloudinary Setup ====
try:
    cloudinary.config(
        cloud_name = CLOUDINARY_CLOUD_NAME,
        api_key = CLOUDINARY_API_KEY,
        api_secret = CLOUDINARY_API_SECRET,
        secure = True # Added secure=True, good practice
    )
    print("Cloudinary configured successfully")
except Exception as e:
    print(f"Failed to configure Cloudinary: {e}")


def upload_image_and_get_url(image_path):
    """Uploads image to Cloudinary and returns the public URL."""
    try:
        response = cloudinary.uploader.upload(image_path)
        return response["secure_url"]
    except Exception as e:
        print(f"Error during Cloudinary upload: {e}")
        raise # Re-raise error so it's caught by the caller

class ImageDescriptionGenerator:
    def __init__(self, api_base=API_BASE, api_key=OPENROUTER_API_KEY, model=MODEL, api_timeout=API_TIMEOUT):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_timeout = api_timeout
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_qwen_response(self, image_url, prompt_text):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=256,
                timeout=self.api_timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"ERROR: API call failed in get_qwen_response: {e}")
            # Return error string, caller needs to handle it
            return f"Error generating description: {e}"

    def generate_description(self, image_path, context=""):
        """Generates alt_text and description, returning a dictionary."""
        # Note: This makes TWO API calls per image if context is different
        try:
            image_url = upload_image_and_get_url(image_path)
            prompt = "Describe this image for alt text."
            alt_text = self.get_qwen_response(image_url, prompt)

            # Generate longer description if needed (uses context if provided)
            desc_prompt = "Describe this image."
            if context:
                desc_prompt += f" Context: {context}"
            long_description = self.get_qwen_response(image_url, desc_prompt)

            return {
                "alt_text": alt_text,
                "description": long_description,
            }
        except Exception as e:
            # Catch errors from upload or if get_qwen_response raises something unexpected
             print(f"ERROR: Failed during image upload or calling get_qwen_response: {e}")
             # Return dict indicating failure
             return {"alt_text": f"Error: {e}", "description": f"Error: {e}"}


class ImageDescriber:
    def __init__(self, device=None, api_base=API_BASE, api_key=OPENROUTER_API_KEY, model=MODEL, api_timeout=API_TIMEOUT):
        self.device = device
        self._generator = ImageDescriptionGenerator(api_base, api_key, model, api_timeout)

    def is_available(self):
        """Checks if the OpenRouter /models endpoint is reachable AND requires auth."""
        try:
            response = requests.get(
                f"{self._generator.api_base}/models",
                headers=self._generator.headers,
                timeout=10
            )
            print(f"DEBUG: is_available check to {response.url} status: {response.status_code}") # Keep this!
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network error during is_available check: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error during is_available check: {e}")
            return False

    def generate_description(self, image_bytes):
        """Generates descriptions and returns the alt_text string."""
        if not image_bytes:
            print("Warning: Empty image bytes provided")
            return None

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            # Call the generator which returns a dictionary
            result = self._generator.generate_description(temp_path) # No context passed

            # Extract and return the alt_text
            # Handle potential errors returned within the dict
            alt_text = result.get("alt_text")
            if alt_text and alt_text.lower().startswith("error:"):
                 print(f"Warning: Received error in alt_text: {alt_text}")
                 return None # Treat error as failure for the caller
            return alt_text # Return the alt_text string or None if key missing

        except Exception as e:
            # Catch errors from tempfile operations or if generate_description raises unexpected error
            print(f"ERROR: Error generating image description in ImageDescriber: {e}")
            return None # Return None on failure
        finally:
            # Robust cleanup
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    print(f"Warning: Failed to delete temporary file {temp_path}: {e}")

def create_descriptor(device=None):
    return ImageDescriber(device=device)

# --- END OF vision_descriptor.py ---