# pipeline_utils.py

"""
Utility functions for the Exam Processing Pipeline.
Includes dependency checks, filesystem operations, and configuration display.
"""

import sys
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional # Added Optional
import json


# Assuming dependencies.py is available for the check function
try:
    import dependencies
except ImportError:
    # Define a dummy function if dependencies.py is missing, preventing immediate crash
    # but allowing the pipeline to report the missing module later if called.
    logging.getLogger(__name__).critical("Module 'dependencies.py' not found. Dependency checking will fail.")
    class dependencies:
        @staticmethod
        def install_and_check(pkgs: List[str], install_if_missing: bool = True) -> bool:
            logging.getLogger(__name__).error("Cannot check dependencies: 'dependencies.py' module missing.")
            return False # Assume unavailable if module is missing

log = logging.getLogger(__name__)

# --- Dependency Checks ---

def check_optional_dependencies(required_optionals: List[str] = ['openpyxl']) -> Dict[str, bool]:
    """
    Checks for optional dependencies using the dependencies module and logs their status.

    Args:
        required_optionals: A list of optional package names to check.

    Returns:
        A dictionary mapping package names to a boolean indicating availability.
    """
    availability = {}
    log.info("Checking optional dependencies...")
    # Ensure the function exists before calling
    if not hasattr(dependencies, 'install_and_check'):
        log.error("Function 'install_and_check' not found in dependencies module. Cannot check optional dependencies.")
        return {pkg: False for pkg in required_optionals}

    for pkg_name in required_optionals:
        # Call with install_if_missing=False to only check availability
        is_available = dependencies.install_and_check([pkg_name], install_if_missing=False)
        if is_available:
            log.info(f"✅ Optional dependency '{pkg_name}' found.")
            availability[pkg_name] = True
        else:
            # The warning is usually logged within install_and_check itself
            log.warning(f"Optional dependency '{pkg_name}' check returned False (likely not installed).")
            availability[pkg_name] = False
    return availability

# --- Filesystem Operations ---

def setup_directories(config: Dict[str, Any]) -> bool:
    """
    Creates input and output directories defined in the config if they don't exist.

    Args:
        config: The pipeline configuration dictionary, expected to contain
                'INPUT_DIR' and 'BASE_OUTPUT_DIR' keys.

    Returns:
        True if directories are ready, False otherwise.
    """
    log.info("Setting up directories...")
    input_dir_str = config.get('INPUT_DIR')
    output_dir_str = config.get('BASE_OUTPUT_DIR')

    if not input_dir_str or not output_dir_str:
        log.error("Configuration incomplete: Missing 'INPUT_DIR' or 'BASE_OUTPUT_DIR'.")
        return False

    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Input directory ready: '{input_dir.resolve()}'")
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory ready: '{output_dir.resolve()}'")
        return True
    except PermissionError:
        log.exception(f"Permission denied creating directories (Input: {input_dir}, Output: {output_dir}). Check filesystem permissions.")
        return False
    except Exception as e:
        log.exception(f"Failed to create directories (Input: {input_dir}, Output: {output_dir}): {e}")
        return False

def find_documents(input_dir_path: Path, supported_extensions: Set[str]) -> List[Path]:
    """
    Scans the input directory for processable documents based on extensions.
    Filters out temporary files (starting with '~').

    Args:
        input_dir_path: The Path object for the input directory.
        supported_extensions: A set of lowercase file extensions (e.g., {'.pdf', '.docx'}).

    Returns:
        A sorted list of Path objects for the documents found. Returns empty list on error.
    """
    log.info(f"Scanning for documents in: '{input_dir_path.resolve()}'")
    documents_to_process = []
    skipped_files = []

    if not supported_extensions:
        log.warning("No supported extensions provided. No documents will be found.")
        return []

    try:
        if not input_dir_path.is_dir():
            log.error(f"Input directory '{input_dir_path}' does not exist or is not a directory.")
            return []

        all_items = list(input_dir_path.iterdir()) # Get all items once
        log.debug(f"Found {len(all_items)} total items in input directory.")

        documents_to_process = sorted([
            p for p in all_items if p.is_file()
            and p.suffix.lower() in supported_extensions
            and not p.name.startswith('~$') # Skip temp Word/Excel/etc. files
        ])

        skipped_files = [
             p.name for p in all_items if p.is_file() and p not in documents_to_process
        ]

        log.info(f"Found {len(documents_to_process)} supported document(s) matching extensions: {', '.join(supported_extensions)}.")
        if skipped_files:
            log.info(f"Skipped {len(skipped_files)} file(s): {', '.join(skipped_files)}")
        if documents_to_process:
            for doc in documents_to_process:
                log.debug(f"  + Will process: {doc.name}")
        else:
            log.warning(f"No documents matching supported extensions found in '{input_dir_path}'.")

    except PermissionError:
        log.exception(f"Permission denied while scanning directory '{input_dir_path}'. Check filesystem permissions.")
        return []
    except Exception as e:
        log.exception(f"Error scanning input directory '{input_dir_path}': {e}")
        return [] # Return empty list on error

    return documents_to_process

# --- Configuration Display ---

def display_configuration(config: Dict[str, Any]):
    """
    Logs the current runtime configuration settings, redacting sensitive keys.

    Args:
        config: The pipeline configuration dictionary.
    """
    log.info("--- Runtime Configuration ---")
    # Define keys to display (adjust as needed based on pipeline_config.py)
    keys_to_display = sorted([
        'INPUT_DIR', 'BASE_OUTPUT_DIR', 'SUPPORTED_EXTENSIONS',
        'DOCLING_DEVICE', 'DOCLING_IMAGE_SCALE', 'DOCLING_BOOK_LIKE_MD_SUFFIX',
        'LLM_API_URL', 'LLM_API_KEY', 'LLM_MODEL_NAME',
        'LLM_REFORMAT_API_URL', 'OPENROUTER_API_KEY', 'LLM_REFORMAT_MODEL_NAME', 'FORMATTED_MD_SUFFIX',
        'LLM_TRANSCRIPT_API_URL', 'LLM_TRANSCRIPT_API_KEY', 'LLM_TRANSCRIPT_MODEL_NAME', 'TRANSCRIPT_JSON_SUFFIX',
        'LLM_IR_API_URL', 'LLM_IR_API_KEY', 'LLM_IR_MODEL_NAME', 'IR_JSON_SUFFIX',
        'LLAVA_API_URL', 'LLAVA_MODEL', 'LLAVA_CONTEXT_CHARS', 'IMAGE_DESC_JSON_SUFFIX',
        'TTS_LANGUAGE', 'TTS_OUTPUT_SUBDIR',
        'VOICE_ENGINE_LANGUAGE', 'VOICE_ENGINE_COMMAND_TIMEOUT', 'VOICE_ENGINE_ANSWER_TIMEOUT', 'VOICE_ENGINE_CONFIRM_TIMEOUT',
        'TIMEOUTS', 'MAX_RETRIES',
        'VALIDATE_IR_SCHEMA', 'ENABLE_TTS', 'LAUNCH_VOICE_ENGINE_AFTER',
        'SKIP_MARKDOWN_REFORMATTING', 'SKIP_IMAGE_DESCRIPTION', 'SKIP_IR_GENERATION', 'SKIP_TRANSCRIPT_GENERATION'
    ])

    # Filter keys that are actually present in the passed config
    available_keys = [k for k in keys_to_display if k in config]
    if not available_keys:
        log.warning("Configuration dictionary provided is empty or contains none of the expected keys.")
        return

    max_key_len = max(len(k) for k in available_keys)

    for key in available_keys:
        value = config[key]
        value_display = value # Default display

        # Securely display sensitive keys
        if "API_KEY" in key.upper() and isinstance(value, str) and value:
            value_display = f"'{value[:4]}...{value[-4:]}'" if len(value) > 8 else "'***'"
        # Resolve paths for better clarity
        elif ('DIR' in key.upper() or 'PATH' in key.upper() or key.upper().endswith('_FILE')) and isinstance(value, (str, Path)) and value:
            try:
                # Use str() to handle Path objects gracefully
                value_display = f"'{Path(str(value)).resolve()}'"
            except Exception: # Handle potential errors if path doesn't exist yet
                value_display = f"'{Path(str(value)).absolute()}' (may not exist yet)"
        # Handle sets nicely
        elif isinstance(value, set):
             value_display = f"{{{', '.join(sorted(list(str(v) for v in value)))}}}" if value else "{}"
        # Add specific formatting for other types if needed (like TIMEOUTS dict)
        elif key == 'TIMEOUTS' and isinstance(value, dict):
             value_display = json.dumps(value) # Display dicts as JSON string


        log.info(f"  {key:<{max_key_len}} : {value_display}")

    log.info("-" * (max_key_len + 15)) # Adjust width as needed