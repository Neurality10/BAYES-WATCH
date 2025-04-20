# config.py
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    # Voice Engine Settings
    VOICE_ENGINE_LANGUAGE = 'en-US'
    VOICE_ENGINE_PAUSE_THRESHOLD = 0.8
    VOICE_ENGINE_ENERGY_THRESHOLD = 400
    VOICE_ENGINE_COMMAND_TIMEOUT = 5
    VOICE_ENGINE_ANSWER_TIMEOUT = 10
    AMBIENT_NOISE_ADJUST_DURATION = 0.8
    
    # UI States
    UI_STATE_SELECTING_EXAM = 'selecting_exam'
    UI_STATE_SHOWING_ITEM = 'showing_item'
    UI_STATE_LISTENING_COMMAND = 'listening_command'
    UI_STATE_LISTENING_ANSWER = 'listening_answer'
    UI_STATE_EDITING_ANSWER = 'editing_answer'
    UI_STATE_SUBMITTING = 'submitting'
    UI_STATE_FINISHED = 'finished'

    @staticmethod
    def get_config_value(key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to default."""
        return getattr(Config, key, default)

    @staticmethod
    def set_session_state(key: str, value: Any) -> None:
        """Safely set Streamlit session state."""
        import streamlit as st
        st.session_state[key] = value

    @staticmethod
    def get_session_state(key: str, default: Any = None) -> Any:
        """Safely get Streamlit session state with default."""
        import streamlit as st
        return st.session_state.get(key, default)

# --- User Configuration ---
# <<< *** MODIFY THESE SETTINGS AS NEEDED *** >>>
CONFIG_OUTPUT_DIR = "processed_docs_output" # Output folder name
# --- OCR DISABLED ---
CONFIG_OCR_ENGINE = None                  # Set to None or '' to disable OCR.
CONFIG_DEVICE = "auto"                    # 'cpu', 'cuda', 'mps', 'auto'
CONFIG_IMAGE_SCALE = 2.0                  # For image extraction quality (must be > 0)
CONFIG_OCR_LANGS = []                     # Not used when OCR is disabled
# --------------------
INPUT_DIR = Path("./input_docs")          # Folder containing input documents
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}  # File extensions to process
# <<< *** END OF USER CONFIGURATION *** >>>

# --- Internal/Derived Configuration ---
# (Add any other constants derived from user config if needed)

# Ensure image scale is valid, provide a default if not.
if not isinstance(CONFIG_IMAGE_SCALE, (int, float)) or CONFIG_IMAGE_SCALE <= 0:
    print(f"WARNING (config.py): Invalid CONFIG_IMAGE_SCALE ({CONFIG_IMAGE_SCALE}). Setting to default 2.0.")
    CONFIG_IMAGE_SCALE = 2.0 
    
    
    
    
    
    
    
    
    
    