# image_descriptor.py
"""
Generates descriptions and alt text for images found in a Markdown file
using a LLaVA model via API.
"""

import requests
import base64
import os
import json
import re
from PIL import Image
import io
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Setup logger for this module
log = logging.getLogger(__name__)

# --- Helper Functions ---

def encode_image(image_path: Path) -> Optional[str]:
    """Loads an image, ensures RGB, and encodes it to Base64."""
    if not image_path.is_file():
        log.error(f"Image file not found at {image_path}")
        return None
    try:
        log.debug(f"Attempting to load and encode image: {image_path}")
        with Image.open(image_path) as img:
            # Ensure image is in RGB format, as required by many models
            if img.mode != 'RGB':
                log.debug(f"Converting image {image_path.name} from {img.mode} to RGB.")
                img = img.convert('RGB')
            # Save image to a bytes buffer
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")  # Using PNG, usually lossless
            # Encode bytes to Base64 string
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
            log.debug(f"Successfully encoded image: {image_path.name}")
            return encoded_string
    except FileNotFoundError: # Should be caught by is_file check, but good to have
        log.error(f"Image file not found during encoding attempt: {image_path}")
        return None
    except Exception as e:
        log.exception(f"Error processing image {image_path}: {e}")
        return None


def get_llava_response(image_base64: str, prompt_text: str, config: Dict[str, Any]) -> str:
    """Sends image and prompt to configured LLaVA API and returns the text response."""
    if not image_base64:
        # Error should have been logged during encoding attempt
        return "Error: Cannot process image (encoding failed or invalid input)."

    # --- Get config values ---
    llava_model = config.get('LLAVA_MODEL')
    llava_api_url = config.get('LLAVA_API_URL')
    timeout = config.get('TIMEOUTS', {}).get('llava', 180)

    if not llava_model or not llava_api_url:
        log.error("LLaVA model name or API URL is not configured.")
        return "Error: LLaVA API not configured."

    payload = {
        "model": llava_model,
        "prompt": prompt_text,
        "images": [image_base64],
        "stream": False  # We want the full response at once
        # Add other Ollama options if needed from config
    }

    try:
        log.info(f"Sending request to LLaVA (Model: {llava_model})...")
        log.debug(f"LLaVA Request Payload (excluding image): { {k:v for k,v in payload.items() if k != 'images'} }")

        response = requests.post(llava_api_url, json=payload, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_json = response.json()
        log.debug(f"LLaVA Raw Response: {response_json}")
        generated_text = response_json.get("response", "").strip()

        if not generated_text:
            log.warning("LLaVA returned an empty response.")
            if "error" in response_json:
                err_msg = response_json['error']
                log.error(f"LLaVA API Error: {err_msg}")
                return f"Error: LLaVA API returned an error: {err_msg}"
            return "Error: LLaVA returned an empty response."

        log.info(f"LLaVA response received.")
        return generated_text

    except requests.exceptions.Timeout:
        log.error(f"Timeout after {timeout}s connecting to LLaVA API ({llava_api_url})")
        return "Error: LLaVA request timed out."
    except requests.exceptions.HTTPError as e:
        log.error(f"LLaVA API request failed with HTTP status {e.response.status_code}.")
        try:
            log.error(f"LLaVA Response body: {e.response.text}")
        except Exception:
            log.error("Could not decode LLaVA error response body.")
        return f"Error: LLaVA API request failed ({e.response.status_code})."
    except requests.exceptions.RequestException as e:
        log.exception(f"Error communicating with LLaVA API: {e}")
        return f"Error: Could not get description from LLaVA ({type(e).__name__})."
    except json.JSONDecodeError:
        log.error(f"Error decoding LLaVA API JSON response: {response.text}")
        return "Error: Could not parse LLaVA response (Invalid JSON)."
    except Exception as e:
        log.exception(f"An unexpected error occurred during LLaVA interaction: {e}")
        return f"Error: An unexpected error occurred: {type(e).__name__}"


def find_context(markdown_content: str, image_tag_start_pos: int, image_tag_end_pos: int, config: Dict[str, Any]) -> str:
    """Extracts text surrounding the image tag from the Markdown content."""
    context_chars = config.get('LLAVA_CONTEXT_CHARS', 500)
    start_index = max(0, image_tag_start_pos - context_chars // 2)
    end_index = min(len(markdown_content), image_tag_end_pos + context_chars // 2)

    context_slice = markdown_content[start_index:end_index]

    # Try to remove the specific image tag from the context snippet itself
    image_tag_in_slice_start = image_tag_start_pos - start_index
    image_tag_in_slice_end = image_tag_end_pos - start_index

    # Ensure indices are valid within the slice
    if 0 <= image_tag_in_slice_start < image_tag_in_slice_end <= len(context_slice):
        context_without_tag = (
            context_slice[:image_tag_in_slice_start]
            + "[IMAGE LOCATION]"  # Replace tag with placeholder
            + context_slice[image_tag_in_slice_end:]
        )
    else:
        # Fallback if tag position calculation seems off - less precise
        image_tag_pattern = re.compile(r'!\[.*?\]\(.*?\)') # Recompile pattern here
        context_without_tag = image_tag_pattern.sub('[IMAGE LOCATION]', context_slice, count=1)
        log.debug("Used regex fallback for removing image tag from context slice.")

    # Basic cleanup - remove excessive whitespace
    cleaned_context = ' '.join(context_without_tag.split())
    return cleaned_context


def _process_single_md_for_images(markdown_file_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Internal function to process one Markdown file. Finds images, extracts context,
    calls LLaVA for descriptions, and returns structured data list.

    Args:
        markdown_file_path: Path to the input markdown file.
        config: The pipeline configuration dictionary.

    Returns:
        A list of dictionaries, each containing data for one image found.
        Returns an empty list if no images are found or errors occur during file reading.
    """
    log.info(f"Processing Markdown file for images: {markdown_file_path.name}")
    image_data_list = []
    # Base directory for resolving relative image paths is the directory of the MD file
    base_dir = markdown_file_path.parent

    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        log.error(f"Markdown file not found: {markdown_file_path}")
        return []
    except Exception as e:
        log.exception(f"Error reading Markdown file {markdown_file_path}: {e}")
        return []

    # Regex to find Markdown image tags: ![alt text](source/path.png)
    image_pattern = re.compile(r'!\[(?P<alt>.*?)\]\((?P<src>.*?)\)')

    found_images = list(image_pattern.finditer(markdown_content))
    if not found_images:
        log.info(f"No image tags found in {markdown_file_path.name}.")
        return []

    log.info(f"Found {len(found_images)} image tag(s) in {markdown_file_path.name}. Processing...")

    for i, match in enumerate(found_images):
        original_alt_text = match.group('alt')
        # Normalize path separators and remove potential leading/trailing whitespace
        src_path_relative = match.group('src').strip().replace('\\', '/')
        match_start = match.start()
        match_end = match.end()

        log.info(f"Processing Image {i + 1}/{len(found_images)}:")
        log.info(f"  Source Path (relative): {src_path_relative}")
        log.info(f"  Original Alt Text: '{original_alt_text}'")

        # --- Resolve Absolute Image Path ---
        # Use resolve() for a canonical absolute path
        try:
            full_image_path = base_dir.joinpath(src_path_relative).resolve()
            log.info(f"  Resolved Full Path: {full_image_path}")
        except Exception as resolve_err:
            log.error(f"  Could not resolve image path '{src_path_relative}' relative to '{base_dir}': {resolve_err}")
            # Add placeholder data or skip this image? Skipping for now.
            continue

        # --- Get Context ---
        context_text = find_context(markdown_content, match_start, match_end, config)
        log.info(f"  Context snippet extracted ({len(context_text)} chars).")
        log.debug(f"  Context: {context_text[:100]}...")

        # --- Encode Image ---
        img_b64 = encode_image(full_image_path)

        # Initialize descriptions with error messages in case LLaVA fails
        alt_text_generated = f"Error processing image: {src_path_relative}"
        description_generated = f"Error processing image: {src_path_relative}"

        if img_b64:
            # --- Generate Detailed Description (Task 1/2) ---
            prompt_detailed = f"""Context:
---
{context_text}
---

Instruction:
1. First, provide a detailed, objective, and literal description of the visual elements in the image provided above, as if explaining it to someone who cannot see it. Describe the layout (e.g., top-to-bottom, left-to-right), shapes, colors, any text labels or captions *exactly* as they appear, lines, arrows, connections, and any other purely visual details. Do NOT interpret the meaning or refer back to the context in this part.
2. After completing the literal visual description, write "--- Interpretation ---" on a new line. Then, briefly explain the apparent purpose or subject of this image, considering the visual elements you just described and the provided context above.
"""
            log.info("  Generating detailed description (Task 1/2)...")
            description_generated = get_llava_response(img_b64, prompt_detailed, config)
            time.sleep(1) # Small delay between API calls

            # --- Generate Concise Alt Text (Task 2/2) ---
            prompt_alt = f"""Context:
---
{context_text}
---

Instruction:
Based ONLY on the image provided and the context, write a very concise summary (ideally under 125 characters, strict maximum 150 characters) suitable for screen reader alt text. Identify the main subject and purpose of the image. Do NOT give a detailed description of elements. Examples: "Muller-Lyer illusion diagram showing two arrows", "Table comparing SPEC integer and floating point instruction mixes".
"""
            log.info("  Generating concise alt text (Task 2/2)...")
            alt_text_generated = get_llava_response(img_b64, prompt_alt, config)

            # --- Basic Cleanup/Validation for Alt Text ---
            if alt_text_generated.startswith("Error:"):
                log.warning(f"Failed to generate alt text from LLaVA, using fallback. Error: {alt_text_generated}")
                alt_text_generated = f"Requires description: {os.path.basename(src_path_relative)}"
            else:
                if len(alt_text_generated) > 150:
                    log.warning("Generated alt text exceeds 150 chars, shortening.")
                    last_space = alt_text_generated[:147].rfind(' ')
                    if last_space != -1: alt_text_generated = alt_text_generated[:last_space] + "..."
                    else: alt_text_generated = alt_text_generated[:147] + "..."
                # Remove potential leading/trailing quotation marks
                alt_text_generated = alt_text_generated.strip('"')

        else:
            # If image encoding failed earlier
            log.warning("Skipping LLaVA calls due to image encoding error.")
            # Keep the initialized error messages for alt/description

        log.info(f"  Generated Alt Text: '{alt_text_generated}'")
        desc_preview = description_generated.split('\n')[0].strip()
        log.info(f"  Generated Description (Preview): {desc_preview[:100]}{'...' if len(desc_preview) > 100 else ''}")

        # Use the relative path (normalized) as the key identifier within the doc
        # Use the resolved absolute path for reference/debugging
        image_data_list.append({
            "markdown_source_file": markdown_file_path.name,
            "image_source_relative": src_path_relative, # Key field for linking
            "image_full_path": str(full_image_path), # For reference
            "original_alt_text": original_alt_text,
            "extracted_context": context_text,
            "alt_text_generated": alt_text_generated,
            "description_generated": description_generated,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z") # Consider UTC: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })

    return image_data_list


# --- Main Orchestrator Entry Point ---
def generate_descriptions_for_doc(markdown_file_path: Path, output_json_path: Path, config: Dict[str, Any]) -> bool:
    """
    Main function called by the orchestrator. Processes a single Markdown file,
    generates image descriptions using LLaVA, and saves them to a JSON file.

    Args:
        markdown_file_path: Path to the input markdown file.
        output_json_path: Path where the output JSON should be saved.
        config: The pipeline configuration dictionary.

    Returns:
        True if descriptions were generated and saved successfully, False otherwise.
    """
    start_time = time.time()
    log.info(f"Starting image description generation for: {markdown_file_path.name}")

    # Process the single markdown file
    image_descriptions = _process_single_md_for_images(markdown_file_path, config)

    if not image_descriptions:
        log.warning(f"No image descriptions generated for {markdown_file_path.name}. Output file will not be created.")
        # Return True because the process didn't fail, just found nothing to do.
        # If finding images is mandatory, return False here.
        return True

    # --- Format and Save Data ---
    # The required format is a dictionary where the key is the markdown filename
    output_data = {
        markdown_file_path.name: image_descriptions
    }

    log.info(f"Saving {len(image_descriptions)} image description(s) to: {output_json_path.name}")
    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - start_time
        log.info(f"Successfully saved image descriptions in {elapsed:.2f}s.")
        return True
    except IOError as e:
        log.exception(f"Error writing output JSON file '{output_json_path}': {e}")
        return False
    except Exception as e:
        log.exception(f"An unexpected error occurred while writing the JSON file: {e}")
        return False

# No `if __name__ == "__main__":` block needed.