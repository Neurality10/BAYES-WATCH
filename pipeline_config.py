# pipeline_config.py
"""
Central configuration file for the Exam Processing Pipeline.

Reads settings primarily from environment variables with sensible defaults.
Use environment variables for sensitive data like API keys.
"""

import os
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# --- Core Path Settings ---
# Use environment variables like PIPELINE_INPUT_DIR or PIPELINE_OUTPUT_DIR to override defaults.
INPUT_DIR = Path(os.getenv("PIPELINE_INPUT_DIR", "./input_docs"))
BASE_OUTPUT_DIR = Path(os.getenv("PIPELINE_OUTPUT_DIR", "./pipeline_output"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx"} # Lowercase file extensions

# --- File Naming Suffixes & Subdirectories ---
DOCLING_BOOK_LIKE_MD_SUFFIX = "_book_like.md"
FORMATTED_MD_SUFFIX = "_formatted_properly.md"
IMAGE_DESC_JSON_SUFFIX = "_image_descriptions.json"
TRANSCRIPT_JSON_SUFFIX = "_transcripts.json"
TTS_OUTPUT_SUBDIR = "audio_gtts" # Subdirectory within each doc's output dir

# --- Docling Settings ---
# Device: 'auto', 'cpu', 'cuda', 'mps'. Env: DOCLING_DEVICE
DOCLING_DEVICE = os.getenv("DOCLING_DEVICE", "auto")
# Image scaling factor. Env: DOCLING_IMAGE_SCALE
DOCLING_IMAGE_SCALE = float(os.getenv("DOCLING_IMAGE_SCALE", "2.0"))

# --- Default LLM Settings (Fallback for Transcript) ---
# Used if task-specific URLs/Models are not set. Env: LLM_API_URL, LLM_MODEL_NAME, LLM_API_KEY
# Example: LLM_API_URL="http://localhost:11434/api/generate" LLM_MODEL_NAME="llama3"
DEFAULT_LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
DEFAULT_LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:latest")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", None) # Required only if the API needs it

# --- LLM Settings (Markdown Reformatting Task) ---
# Recommend using OpenRouter or a powerful local model.
# Env: OPENROUTER_API_KEY, LLM_REFORMAT_MODEL_NAME, LLM_REFORMAT_API_URL
# If OPENROUTER_API_KEY is set, it will use the OpenRouter endpoint by default.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None) # Set this environment variable!
LLM_REFORMAT_MODEL_NAME = os.getenv("LLM_REFORMAT_MODEL_NAME", "qwen/qwen-2.5-coder-32b-instruct:free") # Default OpenRouter model

# Determine API URL for reformatting
_default_openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
if OPENROUTER_API_KEY:
    LLM_REFORMAT_API_URL = os.getenv("LLM_REFORMAT_API_URL", _default_openrouter_url)
else:
    # Fallback to default LLM URL if no OpenRouter key and no specific URL override
    LLM_REFORMAT_API_URL = os.getenv("LLM_REFORMAT_API_URL", DEFAULT_LLM_API_URL)
    if LLM_REFORMAT_API_URL == _default_openrouter_url:
         log.warning("LLM_REFORMAT_API_URL points to OpenRouter, but OPENROUTER_API_KEY is not set. Reformatting might fail.")
    elif LLM_REFORMAT_API_URL == DEFAULT_LLM_API_URL:
         log.info(f"No OpenRouter API key found. Using default LLM API URL ({DEFAULT_LLM_API_URL}) for Markdown Reformatting.")
         # Optional: Adjust model if the default OpenRouter one won't work locally
         # if "/" in LLM_REFORMAT_MODEL_NAME: # Simple check for likely non-local model names
         #      log.warning(f"Reformatting model '{LLM_REFORMAT_MODEL_NAME}' might be OpenRouter-specific. Consider setting LLM_REFORMAT_MODEL_NAME to a local model like '{DEFAULT_LLM_MODEL_NAME}' if using local LLM.")
         #      LLM_REFORMAT_MODEL_NAME = DEFAULT_LLM_MODEL_NAME # Uncomment to force default model

# Optional: Site URL/App Name for OpenRouter headers. Env: YOUR_SITE_URL, YOUR_APP_NAME
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost")
YOUR_APP_NAME = os.getenv("YOUR_APP_NAME", "ExamPipeline")

# --- LLM Settings (Transcript Generation Task) ---
# Defaults to the main LLM settings but can be overridden.
# Env: LLM_TRANSCRIPT_MODEL_NAME, LLM_TRANSCRIPT_API_URL, LLM_TRANSCRIPT_API_KEY
LLM_TRANSCRIPT_MODEL_NAME = os.getenv("LLM_TRANSCRIPT_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)
LLM_TRANSCRIPT_API_URL = os.getenv("LLM_TRANSCRIPT_API_URL", DEFAULT_LLM_API_URL)
LLM_TRANSCRIPT_API_KEY = os.getenv("LLM_TRANSCRIPT_API_KEY", DEFAULT_LLM_API_KEY)

# --- LLaVA Settings (Image Description Task) ---
# Typically points to a local Ollama instance running LLaVA.
# Env: LLAVA_API_URL, LLAVA_MODEL, LLAVA_CONTEXT_CHARS
LLAVA_API_URL = os.getenv("LLAVA_API_URL", "http://localhost:11434/api/generate")
LLAVA_MODEL = os.getenv("LLAVA_MODEL", "llava:latest")
LLAVA_CONTEXT_CHARS = int(os.getenv("LLAVA_CONTEXT_CHARS", "500"))

# --- TTS Settings (gTTS) ---
# Env: TTS_LANGUAGE
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", 'en')

# --- Voice Engine CLI Settings ---
# Env: VOICE_ENGINE_LANGUAGE, VOICE_ENGINE_COMMAND_TIMEOUT, etc.
VOICE_ENGINE_LANGUAGE = os.getenv("VOICE_ENGINE_LANGUAGE", 'en-US')
VOICE_ENGINE_COMMAND_TIMEOUT = int(os.getenv("VOICE_ENGINE_COMMAND_TIMEOUT", "10"))
VOICE_ENGINE_ANSWER_TIMEOUT = int(os.getenv("VOICE_ENGINE_ANSWER_TIMEOUT", "15"))
VOICE_ENGINE_CONFIRM_TIMEOUT = int(os.getenv("VOICE_ENGINE_CONFIRM_TIMEOUT", "8"))

# --- Pipeline Control & Behavior ---
# Timeouts (seconds) for network requests. Override specific keys if needed.
TIMEOUTS = {
    "default": 180,
    "docling": 300,
    "llm_reformat": 300,
    "llava": 180,
    "llm_transcript": 240,
    "tts": 60,
    "tts_delay": 0.5 # Delay between individual gTTS calls
}
# Max retries for failed API calls / steps. Env: MAX_RETRIES
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1")) # Default to 1 retry

# --- Optional Step Skipping Flags ---
# Set corresponding environment variable to "false" to disable skipping (i.e., run the step).
# Example: export SKIP_MARKDOWN_REFORMATTING="false"
SKIP_MARKDOWN_REFORMATTING = os.getenv("SKIP_MARKDOWN_REFORMATTING", "False").lower() == 'true'
SKIP_IMAGE_DESCRIPTION = os.getenv("SKIP_IMAGE_DESCRIPTION", "False").lower() == 'true'
SKIP_TRANSCRIPT_GENERATION = os.getenv("SKIP_TRANSCRIPT_GENERATION", "False").lower() == 'true'
# Dependent flags are handled below in validation

# --- Feature Toggles ---
# Set corresponding environment variable to "false" to disable.
ENABLE_TTS = os.getenv("ENABLE_TTS", "True").lower() == 'true'
LAUNCH_VOICE_ENGINE_AFTER = os.getenv("LAUNCH_VOICE_ENGINE_AFTER", "False").lower() == 'true'

# --- Configuration Validation & Logging ---
if not INPUT_DIR.exists():
    log.warning(f"Input directory does not exist: {INPUT_DIR}")
    log.warning("Attempting to create it.")
    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.error(f"Failed to create input directory {INPUT_DIR}: {e}")
        # Consider exiting if input dir is crucial and cannot be created
        # import sys
        # sys.exit(1)

if DOCLING_IMAGE_SCALE <= 0:
    log.warning(f"DOCLING_IMAGE_SCALE must be > 0. Found: {DOCLING_IMAGE_SCALE}. Resetting to 2.0.")
    DOCLING_IMAGE_SCALE = 2.0

# Check for conflicting skip flags / feature toggles
if SKIP_TRANSCRIPT_GENERATION:
    if ENABLE_TTS:
        log.warning("SKIP_TRANSCRIPT_GENERATION is True, disabling ENABLE_TTS.")
        ENABLE_TTS = False
    if LAUNCH_VOICE_ENGINE_AFTER:
        log.warning("SKIP_TRANSCRIPT_GENERATION is True, disabling LAUNCH_VOICE_ENGINE_AFTER.")
        LAUNCH_VOICE_ENGINE_AFTER = False

# Log skip status for clarity
if SKIP_MARKDOWN_REFORMATTING:
    log.info("Skipping Markdown Reformatting step.")
if SKIP_IMAGE_DESCRIPTION:
    log.info("Skipping Image Description step.")
if SKIP_TRANSCRIPT_GENERATION:
    log.info("Skipping Transcript Generation step (TTS and Voice Engine Launch also disabled).")

# Log key LLM configurations
log.info(f"--- Effective LLM Settings ---")
log.info(f"Markdown Reformatting: Model='{LLM_REFORMAT_MODEL_NAME}', API='{LLM_REFORMAT_API_URL}', Key Provided={'Yes' if OPENROUTER_API_KEY else 'No'}")
log.info(f"Transcript Generation: Model='{LLM_TRANSCRIPT_MODEL_NAME}', API='{LLM_TRANSCRIPT_API_URL}', Key Provided={'Yes' if LLM_TRANSCRIPT_API_KEY else 'No'}")
log.info(f"Image Description (LLaVA): Model='{LLAVA_MODEL}', API='{LLAVA_API_URL}'")
log.info(f"------------------------------")

# --- Helper Function (Used by main_pipeline to pass config) ---
def get_config_dict() -> dict:
    """Returns the current configuration settings as a dictionary."""
    config_dict = {}
    # Iterate over a copy of globals dictionary items
    for key, value in list(globals().items()):
        # Include uppercase variables, excluding known internals/modules
        if key.isupper() and not key.startswith('_') and not callable(value) and type(value).__name__ not in ('module', 'Logger'):
            # Resolve Paths for clarity
            if isinstance(value, Path):
                 try: config_dict[key] = str(value.resolve()) # Use resolved path if possible
                 except Exception: config_dict[key] = str(value.absolute()) # Fallback
            # Mask API Keys
            elif "API_KEY" in key and isinstance(value, str) and value:
                 config_dict[key] = f"***{value[-4:]}" if len(value) > 4 else "***"
            else:
                 config_dict[key] = value # Add other values directly

    # Ensure specific calculated/structured values are included
    config_dict['SUPPORTED_EXTENSIONS'] = SUPPORTED_EXTENSIONS # Set is not automatically included
    config_dict['TIMEOUTS'] = TIMEOUTS # Dictionary is not automatically included
    config_dict['TTS_OUTPUT_SUBDIR'] = TTS_OUTPUT_SUBDIR # Include lowercase setting

    # Add final calculated URL for reformatting for clarity
    config_dict['LLM_REFORMAT_API_URL'] = LLM_REFORMAT_API_URL

    return config_dict