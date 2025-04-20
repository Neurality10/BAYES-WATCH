# --- START OF doc_processor_v2_helpers.py ---

import streamlit as st
import os
import sys
import logging
from pathlib import Path

# --- Logging Setup ---
log = logging.getLogger(__name__)

# --- Feature-Specific Imports (Doc Processor V2) ---
# Try importing V2 modules and set a flag
DOC_V2_MODULES_LOADED = False
try:
    # Assuming these are your V2 helper modules/packages
    from docling_handler import DocumentParserPipeline
    from tts_generator import TTSGenerator
    import vision_descriptor # Assuming V2 version
    import dependencies # V2 version
    from docling.datamodel.base_models import InputFormat # Required by pipeline init
    # Ensure pandas/openpyxl are checked if needed by V2 components
    try:
        import openpyxl
        isOpenpyxlAvailable = True
        log.info("openpyxl found for V2.")
    except ImportError:
        isOpenpyxlAvailable = False
        log.warning("openpyxl not found for V2, XLSX output for tables might be skipped.")
    try:
        import pandas as pd
        log.info("pandas found for V2.")
    except ImportError:
        # Log if pandas is strictly required by V2, otherwise it might be optional
        log.warning("pandas not found. Some V2 features might be affected if it relies on it.")

    DOC_V2_MODULES_LOADED = True
    log.info("Document Processor V2 modules loaded successfully.")
except ImportError as e:
    log.error(f"**Error:** Doc Processor V2 modules failed to import: {e}. V2 Features Disabled.")
    DOC_V2_MODULES_LOADED = False
    # Set dummy flag
    isOpenpyxlAvailable = False
except Exception as e:
    log.error(f"**Error:** Unexpected error during V2 module import: {e}", exc_info=True)
    DOC_V2_MODULES_LOADED = False
    isOpenpyxlAvailable = False

# --- Define V2 dummy classes if loading failed ---
# These allow the main app to run without crashing if V2 is unavailable
if not DOC_V2_MODULES_LOADED:
    class DocumentParserPipeline:
        def __init__(self, *args, **kwargs):
            log.error("Attempted to initialize dummy DocumentParserPipeline.")
            # Raise error or simply log it? Log is safer for app stability.
        def process_document(self, *args, **kwargs):
            log.error("Attempted to call dummy process_document.")
            # Return a value indicating failure, consistent with expected real return type if possible
            # Example: return {'success': False, 'status': 'V2 Modules Not Loaded', 'errors': []}
            return {'success': False, 'status': 'V2 Modules Not Loaded', 'errors': []} # Match expected dict structure
    class TTSGenerator:
        def __init__(self, *args, **kwargs):
            log.error("Attempted to initialize dummy TTSGenerator.")
            self.engine_type = "dummy"
        def process_markdown_to_speech(self, *args, **kwargs):
             log.error("Attempted to call dummy process_markdown_to_speech.")
             return [] # Return empty list as expected
    class vision_descriptor:
        @staticmethod
        def create_descriptor(*a,**kw): return None # Dummy static method
    # Define dummy InputFormat if needed, though it's usually used during init
    class InputFormat:
        PDF = "dummy_pdf"
        DOCX = "dummy_docx"
        # Add other formats if your dummy logic needs them

# --- Configuration (Doc Processor V2) ---
# Base directory relative to this helper file (assuming it's in the main app dir)
_APP_DIR = Path(__file__).resolve().parent
# <<< NOTE: Changed back to relative paths as per user's original code >>>
# These will be relative to where app.py is run
APP2_INPUT_DIR_NAME = "input_docs"
APP2_OUTPUT_DIR_NAME = "processed_docs_output_v2"
APP2_TTS_OUTPUT_DIR_NAME = "tts_output_v2"

# Processing Settings
APP2_DOCLING_DEVICE = "auto"
APP2_IMAGE_SCALE = 1.0
APP2_TTS_ENGINE = "gtts" # Default engine if V2 modules load
APP2_TTS_LANG = 'en'
APP2_TTS_CHUNK_SIZE = 4000
APP2_SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# Cloudinary & OpenRouter Keys (CRITICAL: Use Env Vars!)
# These might be better placed in app.py or a dedicated config.py if used elsewhere
# Keeping them here as they relate to V2 features (vision descriptor, potentially others)
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME") # Set default to None if not found
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
QWEN_MODEL = "qwen/qwen2.5-vl-32b-instruct:free"


# --- Ensure App2 Dirs Exist ---
# This code runs when the module is imported
# It creates directories relative to the execution directory (where app.py is run)
try:
    # Construct full paths relative to the current working directory
    # This assumes the helper is imported by an app running in the project root
    cwd = Path.cwd() # Get current working directory where app.py is likely run
    output_dir = cwd / APP2_OUTPUT_DIR_NAME
    tts_output_dir = cwd / APP2_TTS_OUTPUT_DIR_NAME
    # Note: Input dir is handled dynamically in app.py upload logic

    output_dir.mkdir(parents=True, exist_ok=True)
    tts_output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Doc Processor V2 output directories ensured: {output_dir}, {tts_output_dir}")
except Exception as e:
    log.error(f"Failed to create mandatory Doc Processor V2 output directories: {e}", exc_info=True)


# --- Helper Functions (Doc Processor V2) ---

@st.cache_resource # Cache pipelines for the session
def initialize_doc_processor_v2_pipelines(_config):
    """
    Initializes V2 pipelines using loaded modules and config from main app.
    Returns tuple: (docling_pipe | None, tts_gen | None)
    """
    global DOC_V2_MODULES_LOADED, isOpenpyxlAvailable # Access flags set during import

    if not DOC_V2_MODULES_LOADED:
        log.error("Cannot initialize V2 pipelines: Core V2 modules failed to load.")
        return None, None

    log.info("Initializing Document Processor V2 components...")
    docling_pipe, tts_gen = None, None

    # --- Initialize Docling Pipeline (V2) ---
    try:
        # Ensure InputFormat is available (checked during global import)
        if 'InputFormat' not in globals() or not hasattr(InputFormat, 'PDF'): # Basic check
             log.error("Docling InputFormat class not available. Cannot configure formats.")
             raise NameError("InputFormat not loaded.")

        format_map = { ".pdf": InputFormat.PDF, ".docx": InputFormat.DOCX }
        # Use config passed from app.py
        supported_exts_config = _config.get("APP2_SUPPORTED_EXTENSIONS", APP2_SUPPORTED_EXTENSIONS)
        fmts = [format_map[ext] for ext in supported_exts_config if ext in format_map]
        if not fmts:
            log.warning("No configured extensions map to known InputFormat. Defaulting to PDF/DOCX.")
            fmts = [InputFormat.PDF, InputFormat.DOCX]

        # Use output dir path constructed relative to CWD earlier, passed via config
        out_base = Path(_config.get("APP2_OUTPUT_DIR")) # Get full path from config
        if not out_base.exists(): # Double check existence
             log.warning(f"Output base directory '{out_base}' not found during pipeline init, attempting creation.")
             out_base.mkdir(parents=True, exist_ok=True)

        # Check if DocumentParserPipeline class is available
        if 'DocumentParserPipeline' not in globals() or not callable(DocumentParserPipeline):
             log.error("DocumentParserPipeline class is not available or not callable.")
             raise NameError("DocumentParserPipeline class not loaded correctly.")

        # Initialize with correct V2 arguments (ensure these match your V2 class)
        docling_pipe = DocumentParserPipeline(
             base_output_dir=out_base, # Pass the full Path object
             device=_config.get('APP2_DOCLING_DEVICE', APP2_DOCLING_DEVICE),
             image_scale=_config.get('APP2_IMAGE_SCALE', APP2_IMAGE_SCALE),
             allowed_formats=fmts,
             isOpenpyxlAvailable=isOpenpyxlAvailable # Pass the flag determined during import
        )
        log.info("Docling Parser (V2) initialized successfully.")
    except NameError as ne:
        log.error(f"Docling V2 init failed - Required class missing: {ne}", exc_info=True)
        docling_pipe = None
    except Exception as e:
        log.error("CRITICAL: Failed to initialize Docling pipeline (V2).", exc_info=True)
        docling_pipe = None

    # --- Initialize TTS Generator (V2) ---
    tts_engine_config = _config.get("APP2_TTS_ENGINE", APP2_TTS_ENGINE)
    if tts_engine_config and 'TTSGenerator' in globals() and callable(TTSGenerator):
        try:
            tts_gen = TTSGenerator(
                engine_type=tts_engine_config,
                language=_config.get("APP2_TTS_LANG", APP2_TTS_LANG)
            )
            log.info(f"TTS Generator V2 ({tts_gen.engine_type}) initialized successfully.")
        except Exception as e:
            log.error("Failed to initialize TTS Generator V2.", exc_info=True)
            tts_gen = None
    else:
        if not tts_engine_config: log.info("TTS V2 Engine not configured.")
        if 'TTSGenerator' not in globals() or not callable(TTSGenerator): log.info("TTS V2 skipped: TTSGenerator class not available.")
        tts_gen = None

    return docling_pipe, tts_gen

def get_doc_v2_paths(base_output_dir: Path, tts_base_output_dir: Path, doc_stem: str) -> dict:
    """
    Generates standard paths for V2 outputs based on the *actual* base directories.
    """
    # Ensure input args are Path objects
    base_output_dir = Path(base_output_dir)
    tts_base_output_dir = Path(tts_base_output_dir)

    doc_root = base_output_dir / doc_stem
    tts_root = tts_base_output_dir / doc_stem

    # Define expected output structure based on V2 pipeline components
    paths = {
        "root": doc_root,
        # Subdirectory where the primary docling conversion outputs might go
        "docling_output": doc_root / "docling_raw",
        # Standard name for the main markdown generated by V2's markdown_generator
        "book_like_md": doc_root / f"{doc_stem}_book_like.md",
        # Path for JSON from V2's vision_descriptor (if it creates one)
        "image_desc_json": doc_root / f"{doc_stem}_image_descriptions.json",
        # Path for metadata saved by V2's output_saver or docling_handler
        "processing_metadata": doc_root / f"{doc_stem}_processing_metadata.json",
        # Directory where V2's tts_generator saves audio files
        "tts_audio_dir": tts_root,
        # Add other potential V2 outputs here as needed
    }
    return paths

# --- END OF doc_processor_v2_helpers.py ---