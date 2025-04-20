# --- START OF FILE voice_exam_helpers.py ---

# voice_exam_helpers.py
"""
Contains helper functions, state management, UI rendering logic, and command processing
for the Voice Exam Taker feature integrated into the main application.
"""

import streamlit as st
import os
import io
import glob
import json
import traceback
import time
import re
import requests
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
# --- Feature-Specific Imports ---
# Try importing required libraries, log errors if missing
try:
    from gtts import gTTS, gTTSError
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    st.warning("gTTS library not found. TTS functionality will be limited/disabled.", icon="🔊")

try:
    import speech_recognition as sr
    from streamlit_mic_recorder import mic_recorder
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    st.warning("SpeechRecognition or streamlit_mic_recorder not found. Voice input disabled.", icon="🎙️")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.warning("pydub library not found. Some audio operations might fail.", icon="🎵")

# --- Configuration Access ---
# Use the config dict passed from the main app

# --- State Constants ---
VE_STATE_SELECTING_EXAM = "VE_SELECTING_EXAM"
VE_STATE_TAKING_EXAM = "VE_TAKING_EXAM"
VE_STATE_SUBMITTING = "VE_SUBMITTING"
VE_STATE_FINISHED = "VE_FINISHED"

# --- Command Keywords (copied from command_processor.py) ---
NAV_CMDS = ["next", "next question", "skip", "pass"]
PREV_CMDS = ["previous", "previous question", "prev", "back", "go back"]
READ_Q_CMDS = ["read question", "repeat question", "say question"]
READ_A_CMDS = ["read answer", "play answer", "say answer", "preview answer"]
CLEAR_CMDS = ["clear", "clear answer", "clear field", "erase", "erase text", "delete all", "start over"]
UNDO_CMDS = ["undo", "undo change"]
DELETE_LAST_WORD_CMDS = ["delete last word", "remove last word", "del last word"]
BACKSPACE_CMDS = ["backspace", "bksp", "delete last character"]
SUBMIT_CMDS = ["submit", "submit exam", "finish", "finish exam", "done"]
CONFIRM_SUBMIT_CMDS = ["confirm", "confirm submission", "yes", "submit now"]
CANCEL_SUBMIT_CMDS = ["cancel", "cancel submission", "no", "return", "return to exam"]
SAVE_PROGRESS_CMDS = ["save", "save progress"]
STOP_AUDIO_CMDS = ["stop", "stop audio", "stop reading", "silence", "shut up"]

# --- Logging Setup ---
log = logging.getLogger(__name__) # Use __name__

# --- Helper to Get Config Values Safely ---
def get_config_value(config_dict, key, default=None):
    """Safely gets a value from the passed configuration dictionary."""
    return config_dict.get(key, default)

# --- Path Validation (Called on feature load) ---
def validate_paths(config_dict):
    """Checks if essential directories exist based on config."""
    base_output_dir = get_config_value(config_dict, "VE_PIPELINE_OUTPUT_BASE_DIR")
    submissions_dir = get_config_value(config_dict, "VE_SUBMISSIONS_DIR")
    tts_output_dir = get_config_value(config_dict, "VE_TTS_OUTPUT_DIR")

    valid = True
    if not base_output_dir or not Path(base_output_dir).is_dir():
        st.error(f"Configured Pipeline Output Base Directory not found: '{base_output_dir}'. Exams cannot be loaded.")
        log.error(f"Voice Exam: Pipeline output base directory not found or not configured correctly: {base_output_dir}")
        valid = False

    # Create other dirs if they don't exist
    for dir_path_str, dir_name in [(submissions_dir, "Submissions"), (tts_output_dir, "TTS Output")]:
         if not dir_path_str:
             st.warning(f"Configuration for Voice Exam {dir_name} directory missing.")
             log.warning(f"Voice Exam: {dir_name} directory path missing in config.")
             continue # Don't block if optional dirs are missing config
         try:
             dir_path = Path(dir_path_str)
             dir_path.mkdir(parents=True, exist_ok=True)
             log.info(f"Voice Exam: Ensured {dir_name} directory exists: {dir_path.resolve()}")
         except Exception as e:
             st.error(f"Could not create Voice Exam {dir_name} directory '{dir_path_str}': {e}")
             log.error(f"Voice Exam: Failed to create {dir_name} directory at {dir_path_str}: {e}")
             # Don't necessarily set valid=False, but log error

    return valid

# --- State Management ---
def initialize_voice_exam_state():
    """Sets up the default session state variables for the Voice Exam feature."""
    defaults = {
        've_app_state': VE_STATE_SELECTING_EXAM,
        've_exam_data': None,
        've_exam_name': None,
        've_current_q_index': 0,
        've_answers': {},
        've_undo_stack': {},
        've_exam_start_time': None,
        've_audio_status': "Idle",
        've_last_feedback': "Select an exam to begin.",
        've_current_answer_key': 0,
        've_command_input_key': int(time.time()), # Use timestamp for initial unique key
        've_command_input_value': "",
        've_tts_file_to_play': None,
        've_speech_rec_error': False,
        've_processing_voice_command': False,
        've_available_exams': {},
        've_selected_exam_name': None,
        've_selected_exam_run_path': None,
        've_current_exam_json_path': None,
        've_current_audio_base_path': None,
    }
    initialized_now = False
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            initialized_now = True

    # If state was somehow None, reset it
    if st.session_state.get('ve_app_state') is None:
         st.session_state.ve_app_state = VE_STATE_SELECTING_EXAM
         initialized_now = True

    if initialized_now:
        log.info("Voice Exam: Initialized/reset session state variables.")

def reset_to_selection_state():
    """Resets state variables to start over from exam selection."""
    log.info("Voice Exam: Resetting state to selection...")
    keys_to_reset = [k for k in st.session_state if k.startswith('ve_')]
    for key in keys_to_reset:
        try:
            del st.session_state[key]
        except KeyError:
            pass
    # Re-initialize with defaults
    initialize_voice_exam_state()
    st.session_state.ve_last_feedback = "Exam selection reset. Choose an exam."
    log.info("Voice Exam: Session state reset to selection complete.")

# --- Data Handling (Adapted from data_handler.py) ---

def find_available_exams(config_dict):
    """Scans the configured directory for valid exam run folders."""
    base_path = get_config_value(config_dict, "VE_PIPELINE_OUTPUT_BASE_DIR")
    run_prefix = get_config_value(config_dict, "VE_EXAM_RUN_DIR_PREFIX", "streamlit_run_")
    json_pattern = get_config_value(config_dict, "VE_EXAM_JSON_GLOB_PATTERN", os.path.join("*", "*_transcripts.json"))
    audio_pattern = get_config_value(config_dict, "VE_EXAM_AUDIO_DIR_GLOB_PATTERN", os.path.join("*", "audio_gtts"))

    available_exams = {}
    log.info(f"Voice Exam Find: Scanning base path '{base_path}' for runs starting with '{run_prefix}'")

    if not base_path or not Path(base_path).is_dir():
        st.error(f"Voice Exam base directory not found or configured: {base_path}")
        return {}

    try:
        for item_name in os.listdir(base_path):
            item_path = os.path.join(base_path, item_name)
            if Path(item_path).is_dir() and item_name.startswith(run_prefix):
                log.debug(f"Voice Exam Find: Found potential run folder: {item_name}")
                display_name = item_name.replace(run_prefix, "", 1)

                json_search_path = os.path.join(item_path, json_pattern)
                audio_search_path = os.path.join(item_path, audio_pattern)

                # Use Path.glob for more robust pattern matching
                found_json_files = list(Path(item_path).glob(json_pattern))
                found_audio_dirs = [p for p in Path(item_path).glob(audio_pattern) if p.is_dir()]

                if found_json_files and found_audio_dirs:
                    if len(found_json_files) > 1:
                        log.warning(f"Voice Exam Find: Multiple JSON files found in {item_path}. Using first: {found_json_files[0]}")
                    if len(found_audio_dirs) > 1:
                        log.warning(f"Voice Exam Find: Multiple audio directories found in {item_path}. Using first: {found_audio_dirs[0]}")

                    log.info(f"Voice Exam Find: Valid exam run found: '{display_name}' at path '{item_path}'")
                    available_exams[display_name] = str(Path(item_path)) # Store as string path
                else:
                    if not found_json_files: log.debug(f"Voice Exam Find: Skipping '{item_name}', no JSON found matching '{json_pattern}'")
                    if not found_audio_dirs: log.debug(f"Voice Exam Find: Skipping '{item_name}', no audio dir found matching '{audio_pattern}'")

    except Exception as e:
        st.error(f"Error scanning for exams in '{base_path}': {e}")
        log.exception(f"Voice Exam Find: Exception during scan")
        return {}

    log.info(f"Voice Exam Find: Found {len(available_exams)} valid exams.")
    st.session_state.ve_available_exams = available_exams
    return available_exams

def load_exam_data():
    """Loads exam data from the JSON path stored in session state."""
    filepath = st.session_state.get("ve_current_exam_json_path")
    if not filepath:
         st.error("Error: Voice Exam JSON path not set in session state.")
         log.error("Voice Exam Load: ve_current_exam_json_path not found.")
         st.session_state.ve_exam_data = None
         st.session_state.ve_last_feedback = "Error: Could not determine which exam file to load."
         return False

    st.session_state.ve_last_feedback = f"Loading exam data from {Path(filepath).name}..."
    log.info(f"Voice Exam Load: Loading exam data from: {filepath}...")

    try:
        if not Path(filepath).is_file():
            log.error(f"Voice Exam Load: File not found at path: '{filepath}'")
            st.session_state.ve_exam_data = None
            st.session_state.ve_last_feedback = f"Error: Exam file not found: {Path(filepath).name}"
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
             log.error(f"Voice Exam Load: Expected list in JSON '{filepath}', got {type(data)}.")
             st.session_state.ve_exam_data = None
             st.session_state.ve_last_feedback = "Error: Invalid exam file format (must be a list)."
             return False

        valid_data = []
        required_keys = ['id', 'filename_base', 'transcript'] # Use 'transcript' key
        for i, item in enumerate(data):
             if isinstance(item, dict) and all(key in item for key in required_keys):
                 item['internal_q_index'] = i
                 item['original_text_chunk'] = item.get('transcript') # Map transcript to expected key
                 valid_data.append(item)
             else:
                 log.warning(f"Voice Exam Load: Skipping invalid item at index {i} in {filepath}. Missing keys or not a dict.")

        if not valid_data:
             log.error(f"Voice Exam Load: No valid exam items found in '{filepath}'.")
             st.session_state.ve_exam_data = None
             st.session_state.ve_last_feedback = "Error: No valid questions found in the exam file."
             return False

        try:
            valid_data.sort(key=lambda x: int(x.get('id', x['internal_q_index'])))
            log.debug(f"Voice Exam Load: Sorted exam data by 'id' field.")
        except (ValueError, TypeError) as sort_err:
            log.warning(f"Voice Exam Load: Could not sort by 'id' ({sort_err}), keeping internal index order.")
            valid_data.sort(key=lambda x: x['internal_q_index'])

        st.session_state.ve_exam_data = valid_data
        st.session_state.ve_last_feedback = f"Exam '{st.session_state.ve_exam_name}' loaded ({len(valid_data)} questions)."
        log.info(f"Voice Exam Load: Loaded {len(valid_data)} items for exam '{st.session_state.ve_exam_name}'.")
        return True

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from '{Path(filepath).name}'. Check format. Details: {e}")
        st.session_state.ve_last_feedback = "Error: Invalid JSON format in exam file."
        log.exception(f"Voice Exam Load: JSON Decode Error in '{filepath}'")
        st.session_state.ve_exam_data = None
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred loading exam data: {e}")
        st.session_state.ve_last_feedback = f"Error loading exam data: {e}"
        log.exception(f"Voice Exam Load: Unexpected error loading data from '{filepath}'")
        st.session_state.ve_exam_data = None
        return False

def get_current_exam_item():
    """Returns the data for the current question index."""
    if not st.session_state.get('ve_exam_data'):
        log.debug("Voice Exam Data: get_current_exam_item called with no exam_data.")
        return None
    index = st.session_state.get('ve_current_q_index', 0)
    if 0 <= index < len(st.session_state.ve_exam_data):
        return st.session_state.ve_exam_data[index]
    log.warning(f"Voice Exam Data: Invalid current_q_index: {index} (Total items: {len(st.session_state.ve_exam_data)})")
    return None

def _get_audio_path_for_item(item):
    """Internal helper to get the full audio path for a given item."""
    if not item or 'filename_base' not in item:
        log.warning("Voice Exam AudioPath: Item missing or no 'filename_base' key.")
        return None
    audio_base_path = st.session_state.get("ve_current_audio_base_path")
    if not audio_base_path:
        log.error(f"Voice Exam AudioPath: ve_current_audio_base_path not set.")
        return None
    if not Path(audio_base_path).is_dir():
         log.error(f"Voice Exam AudioPath: Audio base path is not a valid directory: {audio_base_path}")
         return None
    filename_base = item['filename_base']
    audio_filename = filename_base + ".mp3" # Assume MP3
    audio_path = Path(audio_base_path) / audio_filename
    return str(audio_path) # Return as string

def get_current_audio_path():
    """Gets the file path for the current question's pre-recorded audio."""
    item = get_current_exam_item()
    if not item: return None
    audio_path = _get_audio_path_for_item(item)
    if not audio_path: return None
    if Path(audio_path).is_file():
        return audio_path
    else:
        log.warning(f"Voice Exam AudioPath: Audio file not found at expected path: {audio_path}")
        return None

def save_current_answer(answer_text):
    """Updates the answer in session state for the current question index."""
    current_index = st.session_state.get('ve_current_q_index', -1)
    if current_index < 0 or not isinstance(st.session_state.get('ve_answers'), dict):
         log.warning("Voice Exam AnswerSave: Invalid index or answers state.")
         return
    previous_answer = st.session_state.ve_answers.get(current_index, "")
    current_answer_str = str(answer_text) if answer_text is not None else ""
    if current_answer_str != previous_answer:
        add_undo_state(current_index, previous_answer)
        st.session_state.ve_answers[current_index] = current_answer_str

def add_undo_state(index, previous_answer_state):
    """Adds a previous answer state to the undo stack."""
    if not isinstance(st.session_state.get('ve_undo_stack'), dict):
        st.session_state.ve_undo_stack = {}
    if index not in st.session_state.ve_undo_stack:
        st.session_state.ve_undo_stack[index] = []
    if not st.session_state.ve_undo_stack[index] or st.session_state.ve_undo_stack[index][-1] != previous_answer_state:
        max_undo_depth = 10
        if len(st.session_state.ve_undo_stack[index]) >= max_undo_depth:
            st.session_state.ve_undo_stack[index].pop(0)
        st.session_state.ve_undo_stack[index].append(previous_answer_state)

def save_answers_to_file(config_dict, final_save=False):
    """Saves the collected answers to a JSON file."""
    log.info(f"Voice Exam Save: Attempting save. Final save: {final_save}")
    current_answers = st.session_state.get('ve_answers')
    if not current_answers:
         st.warning("No answers recorded yet.")
         st.session_state.ve_last_feedback = "No answers to save."
         return False

    exam_data = st.session_state.get('ve_exam_data', [])
    submission_answers = []
    for q_idx, answer in current_answers.items():
         original_item_info = {}
         if 0 <= q_idx < len(exam_data):
             original_item = exam_data[q_idx]
             original_item_info = {
                 "original_id": original_item.get("id"),
                 "filename_base": original_item.get("filename_base")
             }
         else:
             log.warning(f"Voice Exam Save: Index {q_idx} out of bounds for exam_data (len {len(exam_data)}).")
         item_info = {
             "question_index": q_idx + 1,
             "answer_text": answer if answer is not None else "",
             **original_item_info
         }
         submission_answers.append(item_info)

    submission_answers.sort(key=lambda x: x["question_index"])

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        exam_name_slug = st.session_state.get('ve_exam_name', 'exam').replace(' ', '_').replace('.', '')
        submissions_dir_str = get_config_value(config_dict, "VE_SUBMISSIONS_DIR", "submissions_ve")
        submissions_dir = Path(submissions_dir_str) # Use Path
        submissions_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
        filename = submissions_dir / f"{exam_name_slug}_submission_{timestamp}.json"

        try:
            display_path = filename.relative_to(Path.cwd())
        except ValueError:
            display_path = filename # Show absolute path if not relative to CWD

        start_time_unix = st.session_state.get('ve_exam_start_time')
        start_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time_unix)) if start_time_unix else None

        submission_data = {
            "exam_name": st.session_state.get('ve_exam_name', 'Unknown'),
            "submission_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exam_start_time_utc": start_time_utc,
            "total_questions_in_exam": len(exam_data),
            "questions_answered": len([a for a in current_answers.values() if a and str(a).strip()]),
            "answers": submission_answers
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, indent=2, ensure_ascii=False)

        feedback = f"Progress saved to `{display_path}`"
        if final_save:
            feedback = f"Exam submitted successfully! Saved to `{display_path}`"
            st.session_state.ve_app_state = VE_STATE_FINISHED
            st.session_state.ve_last_feedback = feedback
            log.info(f"Voice Exam Save: Final submission saved to {filename}. Transitioning to FINISHED.")
            return True
        else:
            st.session_state.ve_last_feedback = feedback
            st.success(feedback) # Show success message in main app area too
            log.info(f"Voice Exam Save: Progress saved to {filename}.")
            return True

    except Exception as e:
         feedback = f"Error saving submission file: {e}"
         st.session_state.ve_last_feedback = feedback
         log.exception(f"Voice Exam Save: Failed to save submission")
         st.error(feedback)
         return False


# --- Text-to-Speech (Adapted from text_to_speech.py) ---

@st.cache_data(show_spinner=False) # Caching the bytes is still reasonable
def get_tts_audio_bytes(text, lang='en-US'):
    """Generates TTS audio bytes using gTTS. Returns bytes or None."""
    operation = f"get_tts_audio_bytes(text='{text[:50]}...', lang='{lang}')"
    log.debug(f"Voice Exam TTS BYTES - ENTER: {operation}")
    if not GTTS_AVAILABLE: return None # Check if library loaded
    if not text or not isinstance(text, str) or not text.strip():
        log.debug(f"Voice Exam TTS BYTES - SKIP: Empty or invalid text.")
        return None
    try:
        # Config comes from main app, but we still need a default language here
        tts_lang_config = lang # Prioritize passed lang
        tts_lang_short = tts_lang_config.split('-')[0] if tts_lang_config else 'en'
        log.debug(f"Voice Exam TTS BYTES - INFO: Using language code '{tts_lang_short}' for gTTS.")
        tts = gTTS(text=text, lang=tts_lang_short, slow=False)
        fp = io.BytesIO()
        log.debug(f"Voice Exam TTS BYTES - ACTION: Calling gTTS.write_to_fp()...")
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        log.debug(f"Voice Exam TTS BYTES - RESULT: gTTS.write_to_fp() finished. Bytes: {len(audio_bytes)}")
        if not audio_bytes:
            error_msg = f"TTS Error: gTTS returned 0 bytes (lang: {tts_lang_short})."
            log.error(f"Voice Exam TTS BYTES: {error_msg}")
            st.session_state.ve_last_feedback = error_msg
            return None
        log.debug(f"Voice Exam TTS BYTES - EXIT: {operation} - SUCCESS")
        return audio_bytes
    except gTTSError as ge:
         error_msg = f"TTS Generation Error: {ge}. Lang='{tts_lang_short}'."
         log.exception(f"Voice Exam TTS BYTES: {error_msg}")
         st.session_state.ve_last_feedback = error_msg
         return None
    except requests.exceptions.RequestException as ce:
         error_msg = f"TTS Network Error: {ce}. Check internet."
         log.exception(f"Voice Exam TTS BYTES: {error_msg}")
         st.session_state.ve_last_feedback = error_msg
         return None
    except Exception as e:
        error_msg = f"Unexpected Error generating TTS: {e}"
        log.exception(f"Voice Exam TTS BYTES: {error_msg}")
        st.session_state.ve_last_feedback = error_msg
        return None

def trigger_tts_playback(config_dict, text):
    """Generates TTS, saves locally, stores path in session state."""
    operation = f"trigger_tts_playback(text='{text[:50]}...')"
    log.debug(f"Voice Exam TTS TRIGGER - ENTER: {operation}")

    if 've_tts_file_to_play' in st.session_state:
        del st.session_state['ve_tts_file_to_play']
        log.debug("Voice Exam TTS TRIGGER - Cleared previous TTS path.")

    st.session_state.ve_audio_status = "Processing TTS..."
    st.session_state.ve_last_feedback = f"Generating speech..."

    if not GTTS_AVAILABLE:
        st.session_state.ve_audio_status = "Error: gTTS Missing"
        st.session_state.ve_last_feedback = "Cannot generate speech: gTTS library not found."
        return False
    if not text or not isinstance(text, str) or not text.strip():
        log.debug(f"Voice Exam TTS TRIGGER - SKIP: Empty text.")
        st.session_state.ve_audio_status = "Error: No text"
        st.session_state.ve_last_feedback = "Cannot generate speech for empty text."
        return False

    tts_output_dir_str = get_config_value(config_dict, "VE_TTS_OUTPUT_DIR", "tts_output_ve")
    tts_output_dir = Path(tts_output_dir_str)
    tts_output_dir.mkdir(parents=True, exist_ok=True) # Ensure exists

    output_file_path = None
    success = False
    try:
        tts_lang = get_config_value(config_dict, 'VE_TTS_LANGUAGE', 'en-US')
        tts_lang_short = tts_lang.split('-')[0]
        log.debug(f"Voice Exam TTS TRIGGER - INFO: Using language '{tts_lang_short}'.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text_prefix = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        if not safe_text_prefix: safe_text_prefix = "tts"
        filename = f"{timestamp}_{safe_text_prefix}.mp3"
        output_file_path = tts_output_dir / filename
        log.debug(f"Voice Exam TTS TRIGGER - ACTION: Saving to '{output_file_path}'")

        tts = gTTS(text=text, lang=tts_lang_short, slow=False)
        tts.save(str(output_file_path)) # Needs string path
        log.debug(f"Voice Exam TTS TRIGGER - SUCCESS: gTTS.save() finished.")

        if not output_file_path.exists() or output_file_path.stat().st_size == 0:
             error_msg = f"TTS Error: gTTS saved an empty file. Path: {output_file_path}"
             log.error(f"Voice Exam TTS TRIGGER: {error_msg}")
             st.session_state.ve_last_feedback = error_msg
             st.session_state.ve_audio_status = "Error: Save failed"
             return False

        st.session_state['ve_tts_file_to_play'] = str(output_file_path) # Store string path
        log.debug(f"Voice Exam TTS TRIGGER - STATE: Stored TTS path: {output_file_path}")
        st.session_state.ve_last_feedback = f"🔊 TTS ready: '{text[:30]}...'. Player loading."
        st.session_state.ve_audio_status = "Ready (TTS)"
        success = True

    except gTTSError as ge:
         error_msg = f"TTS Generation/Save Error: {ge}."
         log.exception(f"Voice Exam TTS TRIGGER: {error_msg}")
         st.session_state.ve_last_feedback = error_msg
         st.session_state.ve_audio_status = "Error: Generation"
    except requests.exceptions.RequestException as ce:
         error_msg = f"TTS Network Error: {ce}."
         log.exception(f"Voice Exam TTS TRIGGER: {error_msg}")
         st.session_state.ve_last_feedback = error_msg
         st.session_state.ve_audio_status = "Error: Network"
    except OSError as oe:
         error_msg = f"TTS File System Error: {oe}."
         log.exception(f"Voice Exam TTS TRIGGER: {error_msg}")
         st.session_state.ve_last_feedback = error_msg
         st.session_state.ve_audio_status = "Error: File System"
    except Exception as e:
        error_msg = f"Unexpected TTS Error: {e}"
        log.exception(f"Voice Exam TTS TRIGGER: {error_msg}")
        st.session_state.ve_last_feedback = error_msg
        st.session_state.ve_audio_status = "Error: Processing"

    log.debug(f"Voice Exam TTS TRIGGER - EXIT: {operation} - Returning {success}.")
    return success

def stop_audio_playback():
    """Clears pending TTS file path and updates status."""
    operation = "stop_audio_playback()"
    log.debug(f"Voice Exam AUDIO - ENTER: {operation}")
    status_changed = False

    if 've_tts_file_to_play' in st.session_state:
        log.debug(f"Voice Exam AUDIO - ACTION: Clearing pending TTS path: {st.session_state['ve_tts_file_to_play']}")
        del st.session_state['ve_tts_file_to_play']
        status_changed = True

    current_status = st.session_state.get("ve_audio_status", "Idle")
    if current_status != "Idle":
        log.debug(f"Voice Exam AUDIO - STATE: Changing status from '{current_status}' to 'Idle'.")
        st.session_state.ve_audio_status = "Idle"
        st.session_state.ve_last_feedback = "Stopped audio playback / Cleared pending TTS."
        status_changed = True
    else:
        if status_changed:
             st.session_state.ve_last_feedback = "Cleared pending TTS."
        log.debug(f"Voice Exam AUDIO - STATE: Status already 'Idle'.")

    log.debug(f"Voice Exam AUDIO - EXIT: {operation}")

# --- Speech-to-Text (Adapted from speech_to_text.py) ---

# Cache the recognizer instance
@st.cache_resource
def get_recognizer(config_dict):
    """Initializes and returns a SpeechRecognition Recognizer instance."""
    log.debug("Voice Exam STT INIT - ENTER: get_recognizer()")
    if not SPEECH_RECOGNITION_AVAILABLE:
        log.error("Voice Exam STT INIT: SpeechRecognition library not available.")
        st.session_state.ve_speech_rec_error = True
        st.session_state.ve_last_feedback = "Voice input disabled: Required library missing."
        return None
    try:
        log.debug("Voice Exam STT INIT - Step 1: Initializing sr.Recognizer()...")
        r = sr.Recognizer()
        log.debug("Voice Exam STT INIT - Step 1: Success.")

        pause_threshold = float(get_config_value(config_dict, 'VE_VOICE_ENGINE_PAUSE_THRESHOLD', 0.8))
        dynamic_energy = bool(get_config_value(config_dict, 'VE_VOICE_ENGINE_DYNAMIC_ENERGY', True))
        energy_threshold = int(get_config_value(config_dict, 'VE_VOICE_ENGINE_ENERGY_THRESHOLD', 400))

        log.debug(f"Voice Exam STT INIT - Step 2: Applying config: pause={pause_threshold}, dynamic={dynamic_energy}, energy={energy_threshold}")
        r.pause_threshold = pause_threshold
        r.dynamic_energy_threshold = dynamic_energy
        if not dynamic_energy:
             r.energy_threshold = energy_threshold
        log.debug("Voice Exam STT INIT - Step 2: Success.")

        try: # Try listing mics
            log.debug("Voice Exam STT INIT - Step 3: Listing microphones...")
            mic_names = sr.Microphone.list_microphone_names()
            log.debug(f"Voice Exam STT INIT - Step 3: Mics found ({len(mic_names)}): {mic_names}")
        except Exception as mic_err:
             log.warning(f"Voice Exam STT INIT - Step 3: Error listing mics: {mic_err}")

        try: # Test default mic access
            log.debug("Voice Exam STT INIT - Step 4: Testing default microphone access...")
            with sr.Microphone() as source:
                 log.debug(f"Voice Exam STT INIT - Step 4: Default mic opened (Index: {source.device_index}).")
            log.debug("Voice Exam STT INIT - Step 4: Mic test successful.")
        except Exception as mic_open_err:
             log.error(f"Voice Exam STT INIT - Step 4: FAILED to open default microphone: {mic_open_err}")
             st.session_state.ve_speech_rec_error = True
             st.session_state.ve_last_feedback = f"Voice recognizer init error: Failed to access microphone ({mic_open_err})"
             log.debug("Voice Exam STT INIT - EXIT: get_recognizer() - FAILED (Mic access)")
             return None

        log.info(f"Voice Exam STT INIT - Recognizer configured. Dynamic Energy: {r.dynamic_energy_threshold}, Energy Thresh: {r.energy_threshold}, Pause Thresh: {r.pause_threshold}")
        st.session_state.ve_speech_rec_error = False # Mark success
        log.debug("Voice Exam STT INIT - EXIT: get_recognizer() - SUCCESS")
        return r

    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize voice recognizer: {e}", icon="🎙️")
        log.exception(f"Voice Exam STT INIT: Recognizer initialization failed")
        st.session_state.ve_last_feedback = f"Voice recognizer init error: {e}"
        st.session_state.ve_speech_rec_error = True
        log.debug("Voice Exam STT INIT - EXIT: get_recognizer() - FAILED (Exception)")
        return None

def transcribe_audio(config_dict, audio_info, recognizer):
    """Transcribes audio bytes using the provided recognizer."""
    if not recognizer or st.session_state.get('ve_speech_rec_error', True):
        if st.session_state.get('ve_speech_rec_error', True):
             log.error("Voice Exam STT: transcribe_audio called but recognizer init failed earlier.")
        else:
             log.error("Voice Exam STT: transcribe_audio called but recognizer is None.")
             st.session_state.ve_last_feedback = "Recognizer unavailable."
        return None
    if not PYDUB_AVAILABLE:
         log.error("Voice Exam STT: transcribe_audio called but pydub is unavailable.")
         st.session_state.ve_last_feedback = "Audio processing library missing."
         return None
    if not audio_info or 'bytes' not in audio_info or not audio_info['bytes']:
         st.warning("No audio data received from recorder.")
         st.session_state.ve_last_feedback = "No audio captured."
         log.warning("Voice Exam STT: transcribe_audio called with no audio bytes.")
         return None

    log.debug("Voice Exam STT: Starting audio transcription...")
    st.session_state.ve_last_feedback = "Processing recorded audio..."

    try:
        raw_bytes = audio_info['bytes']
        audio_io = io.BytesIO(raw_bytes)
        log.debug(f"Voice Exam STT: Received {len(raw_bytes)} bytes.")

        try:
            log.debug("Voice Exam STT: Attempting pydub load...")
            sound = AudioSegment.from_file(audio_io)
            log.debug(f"Voice Exam STT: pydub load successful. Frame Rate: {sound.frame_rate}")
        except Exception as pydub_err:
            log.error(f"Voice Exam STT: pydub failed to load audio: {pydub_err}. Check FFmpeg.")
            st.error(f"Error processing audio format: {pydub_err}. Is FFmpeg installed?")
            st.session_state.ve_last_feedback = "Error: Could not process audio format (FFmpeg?)."
            return None

        wav_io = io.BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0)
        log.debug("Voice Exam STT: Audio exported to WAV in memory.")

        with sr.AudioFile(wav_io) as source:
            try:
                log.debug("Voice Exam STT: Recognizer recording audio data...")
                audio_data = recognizer.record(source)
                log.debug("Voice Exam STT: Recognizer recorded audio data.")
            except Exception as record_err:
                 log.error(f"Voice Exam STT: recognizer.record() failed: {record_err}")
                 st.error(f"Error during STT recording phase: {record_err}")
                 st.session_state.ve_last_feedback = "Error during STT recording."
                 return None

        language_code = get_config_value(config_dict, 'VE_VOICE_ENGINE_LANGUAGE', 'en-US')
        log.debug(f"Voice Exam STT: Sending to Google SR (lang={language_code})...")
        text = recognizer.recognize_google(audio_data, language=language_code)
        text = text.strip()

        st.session_state.ve_last_feedback = f"Recognized (STT): \"{text}\""
        log.info(f"Voice Exam STT: Transcription successful: '{text}'")
        return text

    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
        st.session_state.ve_last_feedback = "Couldn't understand audio (STT)."
        log.warning("Voice Exam STT: SR UnknownValueError.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech Recognition service error: {e}")
        st.session_state.ve_last_feedback = f"Speech service error (STT): {e}"
        log.error(f"Voice Exam STT: SR RequestError: {e}")
        return None
    except ImportError:
         st.error("Error: Audio processing library missing/broken.")
         st.session_state.ve_last_feedback = "Audio processing library missing (STT)."
         log.error("Voice Exam STT: pydub/ffmpeg import error.")
         return None
    except Exception as e:
        st.error(f"Error during audio processing or transcription: {e}")
        st.session_state.ve_last_feedback = f"Transcription error (STT): {e}"
        log.exception(f"Voice Exam STT: Unexpected error")
        return None

# --- Command Processing (Adapted from command_processor.py) ---

def process_text_input(config_dict, text_input):
    """Parses text input, executes action, returns True if rerun needed."""
    operation = f"process_text_input('{text_input[:50]}...')"
    log.debug(f"Voice Exam CMD - ENTER: {operation}")

    if not text_input or not isinstance(text_input, str):
        log.debug(f"Voice Exam CMD - SKIP: Invalid input.")
        return False

    command = text_input.strip().lower()
    original_text = text_input.strip()
    rerun_needed = True # Default

    # --- Stop Audio Check (First) ---
    if command in STOP_AUDIO_CMDS:
        log.debug(f"Voice Exam CMD - MATCH: STOP_AUDIO_CMDS ('{command}')")
        stop_audio_playback()
        log.debug(f"Voice Exam CMD - EXIT: {operation} after stop. Returning True.")
        return True

    # --- State-Specific Commands ---
    current_state = st.session_state.ve_app_state
    if current_state == VE_STATE_SUBMITTING:
        log.debug(f"Voice Exam CMD - STATE: SUBMITTING")
        stop_audio_playback() # Stop TTS before confirm/cancel

        if command in CONFIRM_SUBMIT_CMDS:
            log.debug("Voice Exam CMD - MATCH: CONFIRM_SUBMIT_CMDS")
            st.session_state.ve_last_feedback = "Submitting final answers..."
            if not save_answers_to_file(config_dict, final_save=True): # Pass config
                st.session_state.ve_last_feedback = "Error saving submission. Please try again or cancel."
            rerun_needed = True
        elif command in CANCEL_SUBMIT_CMDS:
            log.debug("Voice Exam CMD - MATCH: CANCEL_SUBMIT_CMDS")
            st.session_state.ve_app_state = VE_STATE_TAKING_EXAM
            st.session_state.ve_last_feedback = "Submission cancelled. Returned to exam."
            rerun_needed = True
        else:
            log.debug(f"Voice Exam CMD - INVALID: '{command}' during submission.")
            feedback_msg = f"'{original_text}' invalid during confirmation. Use 'confirm' or 'cancel'."
            st.session_state.ve_last_feedback = feedback_msg
            trigger_tts_playback(config_dict, feedback_msg) # Pass config
            rerun_needed = True

        log.debug(f"Voice Exam CMD - EXIT: {operation} from SUBMITTING. Returning {rerun_needed}.")
        return rerun_needed

    # --- General Commands (TAKING_EXAM) ---
    elif current_state == VE_STATE_TAKING_EXAM:
        log.debug(f"Voice Exam CMD - STATE: TAKING_EXAM")
        # Stop audio *before* non-reading commands.
        if command not in READ_Q_CMDS and command not in READ_A_CMDS:
             log.debug("Voice Exam CMD - ACTION: Stopping pending TTS.")
             stop_audio_playback()

        current_q_index = st.session_state.ve_current_q_index
        total_questions = len(st.session_state.ve_exam_data) if st.session_state.ve_exam_data else 0
        current_answer = st.session_state.ve_answers.get(current_q_index, "")

        # Navigation
        if command in NAV_CMDS:
            log.debug("Voice Exam CMD - MATCH: NAV_CMDS")
            if current_q_index < total_questions - 1:
                st.session_state.ve_current_q_index += 1
                st.session_state.ve_current_answer_key += 1
                st.session_state.ve_last_feedback = f"Moved to Question {st.session_state.ve_current_q_index + 1}"
            else: st.session_state.ve_last_feedback = "Already at last question."; rerun_needed = False
        elif command in PREV_CMDS:
            log.debug("Voice Exam CMD - MATCH: PREV_CMDS")
            if current_q_index > 0:
                st.session_state.ve_current_q_index -= 1
                st.session_state.ve_current_answer_key += 1
                st.session_state.ve_last_feedback = f"Moved to Question {st.session_state.ve_current_q_index + 1}"
            else: st.session_state.ve_last_feedback = "Already at first question."; rerun_needed = False

        # TTS Reading
        elif command in READ_Q_CMDS:
            log.debug("Voice Exam CMD - MATCH: READ_Q_CMDS")
            stop_audio_playback()
            item = get_current_exam_item()
            text_to_read = item.get('original_text_chunk', '') if item else ''
            log.debug(f"Voice Exam CMD - INFO: Reading question: '{text_to_read[:50]}...'")
            if text_to_read: trigger_tts_playback(config_dict, text_to_read) # Pass config
            else: trigger_tts_playback(config_dict, "No question text found.") # Pass config
        elif command in READ_A_CMDS:
            log.debug("Voice Exam CMD - MATCH: READ_A_CMDS")
            stop_audio_playback()
            current_answer_to_read = st.session_state.ve_answers.get(current_q_index, "").strip()
            log.debug(f"Voice Exam CMD - INFO: Reading answer: '{current_answer_to_read[:50]}...'")
            if current_answer_to_read: trigger_tts_playback(config_dict, current_answer_to_read) # Pass config
            else: trigger_tts_playback(config_dict, "No answer recorded yet.") # Pass config

        # Answer Manipulation
        elif command in CLEAR_CMDS:
            log.debug("Voice Exam CMD - MATCH: CLEAR_CMDS")
            if current_answer:
                add_undo_state(current_q_index, current_answer)
                st.session_state.ve_answers[current_q_index] = ""
                st.session_state.ve_last_feedback = "Answer cleared."
                st.session_state.ve_current_answer_key += 1
            else: st.session_state.ve_last_feedback = "Answer already empty."; rerun_needed = False
        elif command in UNDO_CMDS:
            log.debug("Voice Exam CMD - MATCH: UNDO_CMDS")
            if current_q_index in st.session_state.ve_undo_stack and st.session_state.ve_undo_stack[current_q_index]:
                restored_answer = st.session_state.ve_undo_stack[current_q_index].pop()
                st.session_state.ve_answers[current_q_index] = restored_answer
                st.session_state.ve_last_feedback = "Undo successful."
                st.session_state.ve_current_answer_key += 1
            else: st.session_state.ve_last_feedback = "Nothing to undo."; rerun_needed = False
        elif command in DELETE_LAST_WORD_CMDS:
            log.debug("Voice Exam CMD - MATCH: DELETE_LAST_WORD_CMDS")
            if current_answer:
                add_undo_state(current_q_index, current_answer)
                words = current_answer.strip().split()
                if words:
                    words.pop()
                    new_answer = ' '.join(words)
                    if new_answer: new_answer += ' '
                    st.session_state.ve_answers[current_q_index] = new_answer
                    st.session_state.ve_last_feedback = "Deleted last word."
                else:
                    st.session_state.ve_answers[current_q_index] = ""
                    st.session_state.ve_last_feedback = "Answer cleared."
                st.session_state.ve_current_answer_key += 1
            else: st.session_state.ve_last_feedback = "Answer empty."; rerun_needed = False
        elif command in BACKSPACE_CMDS:
            log.debug("Voice Exam CMD - MATCH: BACKSPACE_CMDS")
            if current_answer:
                add_undo_state(current_q_index, current_answer)
                st.session_state.ve_answers[current_q_index] = current_answer[:-1]
                st.session_state.ve_last_feedback = "Backspace applied."
                st.session_state.ve_current_answer_key += 1
            else: st.session_state.ve_last_feedback = "Answer empty."; rerun_needed = False

        # Submission / Saving
        elif command in SUBMIT_CMDS:
            log.debug("Voice Exam CMD - MATCH: SUBMIT_CMDS")
            stop_audio_playback()
            st.session_state.ve_app_state = VE_STATE_SUBMITTING
            st.session_state.ve_last_feedback = "Please confirm submission."
        elif command in SAVE_PROGRESS_CMDS:
            log.debug("Voice Exam CMD - MATCH: SAVE_PROGRESS_CMDS")
            stop_audio_playback()
            save_answers_to_file(config_dict, final_save=False) # Pass config
            # Feedback handled internally by save_answers_to_file

        # Go To Item Command
        elif (match := re.match(r"go to (?:item|question|number)\s+(\d+)", command)):
            log.debug("Voice Exam CMD - MATCH: GO_TO_ITEM")
            num_str = match.group(1)
            try:
                num = int(num_str)
                target_idx = num - 1
                if 0 <= target_idx < total_questions:
                    if current_q_index != target_idx:
                        st.session_state.ve_current_q_index = target_idx
                        st.session_state.ve_current_answer_key += 1
                        st.session_state.ve_last_feedback = f"Moved to Question {num}."
                    else: st.session_state.ve_last_feedback = f"Already at Q {num}."; rerun_needed = False
                else:
                    msg = f"Invalid question number: {num}. Max is {total_questions}."
                    st.session_state.ve_last_feedback = msg
                    trigger_tts_playback(config_dict, msg) # Pass config
            except ValueError:
                msg = f"Invalid question number: '{num_str}'."
                st.session_state.ve_last_feedback = msg
                trigger_tts_playback(config_dict, msg) # Pass config

        # Default: Append Text
        else:
            log.debug(f"Voice Exam CMD - DEFAULT: Append text '{original_text[:50]}...'")
            add_undo_state(current_q_index, current_answer)
            separator = ' ' if current_answer and not current_answer.endswith(' ') else ''
            new_answer = current_answer + separator + original_text
            st.session_state.ve_answers[current_q_index] = new_answer
            st.session_state.ve_last_feedback = f"Appended text."
            st.session_state.ve_current_answer_key += 1

        log.debug(f"Voice Exam CMD - EXIT: {operation} from TAKING_EXAM. Returning {rerun_needed}.")
        return rerun_needed

    # --- Handle commands if not TAKING_EXAM or SUBMITTING ---
    else:
        log.debug(f"Voice Exam CMD - STATE: Command '{command}' ignored in state {current_state}.")
        st.session_state.ve_last_feedback = f"Command '{original_text}' ignored."
        rerun_needed = True # Rerun to show feedback
        log.debug(f"Voice Exam CMD - EXIT: {operation} from state {current_state}. Returning {rerun_needed}.")
        return rerun_needed


# --- UI Elements (Adapted from ui_elements.py) ---

def display_status_sidebar(config_dict):
    """Displays status indicators, feedback, and input in the sidebar."""
    with st.sidebar: # Target the main app's sidebar
        st.header("Voice Exam Status")

        # Timer
        start_time = st.session_state.get('ve_exam_start_time')
        current_app_state = st.session_state.get('ve_app_state', VE_STATE_SELECTING_EXAM)
        if start_time and current_app_state not in [VE_STATE_FINISHED, VE_STATE_SELECTING_EXAM]:
            duration_seconds = get_config_value(config_dict, 'VE_EXAM_DURATION_MINUTES', 60) * 60
            elapsed_time = time.time() - start_time
            remaining_time = max(0, duration_seconds - elapsed_time)
            timer_text = "Time's Up!" if remaining_time <= 0 else f"{int(remaining_time // 60):02d}:{int(remaining_time % 60):02d}"
            is_urgent = (0 < remaining_time < 300) or (remaining_time <= 0)
            color = "red" if is_urgent else "inherit"
            st.markdown(f"<p style='font-size: 1.2em; color: {color};'>⏳ Time Left: {timer_text}</p>", unsafe_allow_html=True)
        elif current_app_state == VE_STATE_FINISHED:
             st.markdown("<p style='font-size: 1.2em;'>Time Left: Exam Finished</p>", unsafe_allow_html=True)
        else:
             st.markdown("<p style='font-size: 1.2em;'>Time Left: --:--</p>", unsafe_allow_html=True)
        st.divider()

        # Speaker Status (TTS)
        audio_status = st.session_state.get("ve_audio_status", "Idle")
        audio_icon = "▶️" if "Ready" in audio_status else "⏳" if "Processing" in audio_status else "⚠️" if "Error" in audio_status else "✔️"
        st.metric("TTS Status", f"{audio_status} {audio_icon}")
        st.divider()

        # Feedback Area
        st.subheader("Feedback")
        feedback = st.session_state.get('ve_last_feedback', "Initializing...")
        st.info(feedback)
        st.divider()

        # --- Inputs (only when Taking/Submitting) ---
        input_disabled = current_app_state not in [VE_STATE_TAKING_EXAM, VE_STATE_SUBMITTING]
        if not input_disabled:
            # Text Input
            st.subheader("Command / Dictate (Text)")
            command_key = f"ve_command_input_{st.session_state.get('ve_command_input_key', 0)}"
            user_input = st.text_input(
                "Type command or text, press Enter:", key=command_key,
                placeholder="e.g., next, read question, clear...",
                label_visibility="collapsed", disabled=input_disabled, value=""
            )
            if user_input:
                if not st.session_state.get('ve_processing_voice_command', False):
                    log.debug(f"Voice Exam UI: Text input submitted: '{user_input}'")
                    st.session_state.ve_command_input_value = user_input
                    st.session_state.ve_command_input_key += 1 # Force clear on next rerun
                    # Need to trigger rerun from main loop
                else:
                    log.debug("Voice Exam UI: Text input ignored (processing voice).")

            # Voice Input
            st.subheader("Command / Dictate (Voice)")
            stt_generally_disabled = input_disabled or not SPEECH_RECOGNITION_AVAILABLE
            recognizer = get_recognizer(config_dict) # Pass config
            if recognizer is None or st.session_state.get('ve_speech_rec_error'):
                error_detail = st.session_state.get('ve_last_feedback', "Unknown init error.")
                if "Voice recognizer init error:" in error_detail:
                    error_detail = error_detail.split(":", 1)[1].strip()
                st.warning(f"Speech recognizer unavailable. Voice input disabled.\nReason: {error_detail}", icon="🎙️")
                stt_specifically_disabled = True
            else:
                stt_specifically_disabled = False
            stt_disabled = stt_generally_disabled or stt_specifically_disabled

            # Check if mic_recorder is available
            if 'mic_recorder' not in globals():
                 st.error("streamlit_mic_recorder component not available.")
                 stt_disabled = True

            if not stt_disabled:
                mic_key = f"ve_mic_recorder_{st.session_state.get('ve_command_input_key', 0)}"
                audio_info = mic_recorder(
                    start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording",
                    key=mic_key, use_container_width=True, format="wav"
                )
                if audio_info and not st.session_state.get('ve_processing_voice_command', False):
                    log.debug(f"Voice Exam UI: Mic data received (Bytes: {len(audio_info.get('bytes', 0))}). Transcribing...")
                    st.session_state.ve_last_feedback = "Audio recorded, transcribing..."
                    transcribed_text = transcribe_audio(config_dict, audio_info, recognizer) # Pass config
                    if transcribed_text:
                        log.debug(f"Voice Exam UI: Transcription successful: '{transcribed_text}'.")
                        st.session_state.ve_processing_voice_command = True # Set flag BEFORE setting value
                        st.session_state.ve_command_input_value = transcribed_text
                        st.session_state.ve_command_input_key += 1 # Increment key AFTER successful transcription
                        log.debug(f"Voice Exam UI: Set command, flag=True, key={st.session_state.ve_command_input_key}. Rerunning.")
                        st.rerun() # Rerun immediately to process the command
                    else:
                        log.debug("Voice Exam UI: Transcription failed or no text.")
                        # Feedback already set by transcribe_audio on failure
                elif audio_info:
                    log.debug("Voice Exam UI: Mic data received, but already processing voice command. Ignored.")
            else:
                st.button("🎤 Voice Input Disabled", disabled=True, use_container_width=True)


            # Command List Expander
            with st.expander("Show Available Commands"):
                 st.markdown("""
                 *   Use the **Text Input** OR the **Voice Recorder** above.
                 *   Recognized commands are executed.
                 *   Other spoken/typed text is appended to the current answer.
                 **Navigation:** `next`, `previous`, `go to item <number>`
                 **Reading/Playback:** `read question`, `read answer`, `stop`
                 **Answer Editing:** `(Type/Speak text)`, `clear`, `delete last word`, `backspace`, `undo`
                 **Exam Control:** `save`, `submit`
                 **Submission:** `confirm`, `cancel`
                 """)
            if current_app_state == VE_STATE_SUBMITTING:
                st.caption("Submit confirmation active: Use `confirm` or `cancel`")
        else:
             st.caption("Input controls disabled.")

def display_selection_ui(config_dict):
    """Displays the initial exam selection screen."""
    st.title("Voice Exam Taker")
    st.markdown("Select an exam run generated by the document processing pipeline.")

    # Ensure available exams are loaded if not already in state
    if not st.session_state.get('ve_available_exams'):
         find_available_exams(config_dict) # Attempt to find exams now

    available_exams = st.session_state.get('ve_available_exams', {})
    exam_names = list(available_exams.keys())

    if not exam_names:
         base_dir = get_config_value(config_dict, 'VE_PIPELINE_OUTPUT_BASE_DIR')
         prefix = get_config_value(config_dict, 'VE_EXAM_RUN_DIR_PREFIX')
         st.error(f"No valid exam runs found in: `{base_dir}`.")
         st.info(f"Please ensure the directory exists, is configured correctly in app3.py (APP_CONFIG['VE_PIPELINE_OUTPUT_BASE_DIR']), and contains folders starting with '{prefix}' with the correct internal structure (JSON transcript + audio folder).")
         # Optional: Add a button to re-scan?
         # if st.button("Re-scan for Exams"):
         #     st.session_state.pop('ve_available_exams', None) # Clear cache
         #     st.rerun()
         st.stop()

    # --- FIX IS HERE ---
    # Use the CORRECT prefixed session state key 've_selected_exam_name'
    current_selection = st.session_state.get('ve_selected_exam_name') # Get current value safely
    default_index = exam_names.index(current_selection) if current_selection in exam_names else 0
    # --- END FIX AREA ---

    selected_display_name = st.selectbox(
        "Select Exam:", options=exam_names, key="ve_exam_selector",
        # Use the calculated default_index
        index=default_index
    )

    # --- FIX IS HERE ---
    # Update the session state using the CORRECT prefixed key
    st.session_state.ve_selected_exam_name = selected_display_name
    # --- END FIX AREA ---


    if st.button(f"Load Exam: {selected_display_name}", type="primary", key="ve_load_exam_button"):
        log.debug(f"Voice Exam UI: Load button clicked for: {selected_display_name}")
        selected_run_path = available_exams.get(selected_display_name)
        if not selected_run_path:
            st.error("Error: Could not find path for selected exam.")
            return

        st.session_state.ve_selected_exam_run_path = selected_run_path
        st.session_state.ve_exam_name = selected_display_name # Keep exam_name for consistency? Or use ve_? Use ve_
        st.session_state.ve_exam_name = selected_display_name

        # Find JSON and Audio paths within the selected run path
        # Ensure patterns are strings before joining/globbing
        json_pattern_part = str(get_config_value(config_dict, "VE_EXAM_JSON_GLOB_PATTERN", os.path.join("*", "*_transcripts.json")))
        audio_pattern_part = str(get_config_value(config_dict, "VE_EXAM_AUDIO_DIR_GLOB_PATTERN", os.path.join("*", "audio_gtts")))

        found_json_files = list(Path(selected_run_path).glob(json_pattern_part))
        found_audio_dirs = [p for p in Path(selected_run_path).glob(audio_pattern_part) if p.is_dir()]

        if not found_json_files:
            st.error(f"Could not find transcript JSON file in {selected_run_path} (Pattern: {json_pattern_part})")
            return
        if not found_audio_dirs:
             st.error(f"Could not find audio directory in {selected_run_path} (Pattern: {audio_pattern_part})")
             return

        st.session_state.ve_current_exam_json_path = str(found_json_files[0])
        st.session_state.ve_current_audio_base_path = str(found_audio_dirs[0])
        log.debug(f"Voice Exam UI: Set JSON path: {st.session_state.ve_current_exam_json_path}")
        log.debug(f"Voice Exam UI: Set Audio path: {st.session_state.ve_current_audio_base_path}")

        # Pre-initialize recognizer here to catch mic issues early
        recognizer_init = get_recognizer(config_dict) # Pass config
        if recognizer_init is None:
             st.error("Failed to initialize voice recognizer. Voice input will be unavailable.")
             # Proceed anyway, but voice won't work

        if load_exam_data(): # Uses paths set above
            st.session_state.ve_app_state = VE_STATE_TAKING_EXAM
            st.session_state.ve_exam_start_time = time.time()
            st.session_state.ve_current_q_index = 0
            st.session_state.ve_answers = {}
            st.session_state.ve_undo_stack = {}
            if 've_tts_file_to_play' in st.session_state: del st.session_state['ve_tts_file_to_play']
            st.session_state.ve_last_feedback = f"Exam '{selected_display_name}' started!"
            st.session_state.ve_current_answer_key = 1
            st.session_state.ve_command_input_key += 1
            st.session_state.ve_command_input_value = ""
            st.session_state.ve_processing_voice_command = False
            log.info(f"Voice Exam UI: Exam '{selected_display_name}' loaded. Transitioning to TAKING_EXAM.")
            st.rerun()
        else:
             st.error("Failed to load exam data. Check file integrity.")

    st.markdown("---")
    st.subheader("Instructions")
    st.info("""
    *   Select an exam run generated by the pipeline.
    *   **Listen (Question):** Click ▶️ below the question for pre-recorded audio.
    *   **Listen (TTS):** Use `read question`/`read answer` commands (sidebar). A player will appear here.
    *   **Answer:** Type directly into the **'Your Answer'** box below.
    *   **Use Sidebar for Commands & Dictation.**
    *   **Submit:** Use `submit` command or 'Finish Exam' button.
    """)


def display_exam_item(item, index, total_items):
    """Displays question text and pre-recorded audio player."""
    if not item: return
    st.subheader(f"Question {index + 1} of {total_items}")
    audio_path = get_current_audio_path()
    if audio_path:
        try:
            log.debug(f"Voice Exam UI: Displaying audio player for Q{index + 1}: {audio_path}")
            st.audio(audio_path, format="audio/mpeg") # Assume mp3 from gTTS
        except FileNotFoundError:
             log.error(f"Voice Exam UI: Audio file not found by st.audio: {audio_path}")
             st.caption("(Audio file missing)")
        except Exception as e:
             log.error(f"Voice Exam UI: Failed to display st.audio for {audio_path}: {e}")
             st.caption(f"(Error loading audio)")
    else:
        log.debug(f"Voice Exam UI: No pre-recorded audio path for Q{index + 1}")
        st.caption("(No pre-recorded audio. Use 'read question'.)")
    # Use 'original_text_chunk' which was mapped from 'transcript' during load
    question_text = item.get('original_text_chunk', '*No question text found.*')
    st.markdown(f"> {question_text}")

def display_answer_area():
    """Displays the text area for the current answer."""
    current_index = st.session_state.ve_current_q_index
    # Include index AND answer_key in the Streamlit key
    answer_key = f"ve_answer_area_{current_index}_{st.session_state.ve_current_answer_key}"
    current_answer_text = st.session_state.ve_answers.get(current_index, "")
    label = "Your Answer (Changes saved automatically):"

    def _save_typed_answer():
        # Callback to save answer when text_area changes
        if answer_key in st.session_state:
            new_value = st.session_state[answer_key]
            # Only save if it's different from the current state to avoid loop/excess undo
            if st.session_state.ve_answers.get(current_index, "") != new_value:
                 log.debug(f"Voice Exam UI: Answer area changed for Q{current_index+1}. Saving.")
                 save_current_answer(new_value) # Updates ve_answers and adds undo
        else:
             log.warning(f"Voice Exam UI: Answer key '{answer_key}' not found in callback.")

    st.text_area(
        label, value=current_answer_text, height=300, key=answer_key,
        placeholder="Type your full answer here. Use sidebar commands/dictation.",
        on_change=_save_typed_answer, label_visibility="visible"
    )

def display_pending_tts_audio():
    """Checks session state for TTS file path and displays player."""
    tts_file = st.session_state.get('ve_tts_file_to_play')
    if tts_file and Path(tts_file).exists():
        log.debug(f"Voice Exam UI: Found pending TTS file: {tts_file}.")
        try:
            with st.container():
                st.write("---")
                st.subheader("Text-to-Speech Output")
                st.audio(tts_file, format='audio/mp3')
                st.caption(f"(Playing TTS: {Path(tts_file).name})")
                st.write("---")
        except FileNotFoundError:
            log.error(f"Voice Exam UI: Pending TTS file disappeared: {tts_file}")
            st.warning("Generated TTS audio not found.")
        except Exception as e:
            log.error(f"Voice Exam UI: Failed to display TTS audio {tts_file}: {e}")
            st.error(f"Error loading TTS player: {e}")
        # Clear path AFTER attempting display
        del st.session_state['ve_tts_file_to_play']
        log.debug("Voice Exam UI: Cleared pending TTS path.")
    elif tts_file:
        log.warning(f"Voice Exam UI: Pending TTS path found ({tts_file}), but file doesn't exist.")
        st.warning("Could not find generated TTS audio.")
        del st.session_state['ve_tts_file_to_play']

def display_navigation_buttons():
    """Displays Previous/Next buttons."""
    current_index = st.session_state.ve_current_q_index
    total_questions = len(st.session_state.ve_exam_data) if st.session_state.ve_exam_data else 0
    nav_cols = st.columns(2)
    with nav_cols[0]:
        prev_disabled = (current_index == 0)
        if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True, key="ve_btn_prev"):
             log.debug("Voice Exam UI: Previous button clicked.")
             stop_audio_playback()
             st.session_state.ve_command_input_value = PREV_CMDS[0]
             st.rerun()
    with nav_cols[1]:
        next_disabled = (current_index >= total_questions - 1)
        if st.button("Next ➡️", disabled=next_disabled, use_container_width=True, key="ve_btn_next"):
            log.debug("Voice Exam UI: Next button clicked.")
            stop_audio_playback()
            st.session_state.ve_command_input_value = NAV_CMDS[0]
            st.rerun()

def display_action_buttons(config_dict):
    """Displays Save Progress, Finish buttons."""
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("💾 Save Progress", use_container_width=True, key="ve_btn_save"):
             log.debug("Voice Exam UI: Save Progress button clicked.")
             stop_audio_playback()
             # Trigger save directly here instead of command for simplicity? No, keep command flow.
             st.session_state.ve_command_input_value = SAVE_PROGRESS_CMDS[0]
             st.rerun()
    with action_cols[1]:
        if st.button("🏁 Finish Exam", type="secondary", use_container_width=True, key="ve_btn_finish"):
             log.debug("Voice Exam UI: Finish Exam button clicked.")
             stop_audio_playback()
             st.session_state.ve_command_input_value = SUBMIT_CMDS[0]
             st.rerun()

def display_test_command_buttons(config_dict):
    """Displays test buttons in an expander."""
    st.write("---")
    with st.expander("🧪 TESTING: Trigger Commands Manually", expanded=False):
        current_app_state = st.session_state.get('ve_app_state', VE_STATE_SELECTING_EXAM)
        log.debug(f"Voice Exam TEST UI: State = {current_app_state}")

        def create_button_row(label, commands, cols=3, disable_logic=None, stop_audio=True):
            st.write(f"**{label}:**")
            columns = st.columns(cols)
            col_idx = 0
            for cmd in commands:
                with columns[col_idx % cols]:
                    # Determine disabled state based on logic string
                    disabled = False
                    is_processing_voice = st.session_state.get('ve_processing_voice_command', False)
                    if disable_logic == "not_taking":
                        disabled = (current_app_state != VE_STATE_TAKING_EXAM) or is_processing_voice
                    elif disable_logic == "not_submitting":
                         disabled = (current_app_state != VE_STATE_SUBMITTING) or is_processing_voice
                    elif disable_logic == "always_enabled":
                         disabled = is_processing_voice # Only disable if voice processing
                    else: # Default (taking or submitting allowed)
                         disabled = (current_app_state not in [VE_STATE_TAKING_EXAM, VE_STATE_SUBMITTING]) or is_processing_voice

                    if st.button(cmd.title(), key=f"ve_test_btn_{cmd.replace(' ', '_')}", disabled=disabled, use_container_width=True):
                        log.debug(f"Voice Exam TEST UI: Button '{cmd}' clicked.")
                        if stop_audio: stop_audio_playback()
                        st.session_state.ve_command_input_value = cmd
                        st.rerun()
                col_idx += 1
            st.write("---")

        # Group buttons logically
        create_button_row("Navigation", [NAV_CMDS[0], PREV_CMDS[0]], cols=2, disable_logic="not_taking", stop_audio=True)
        create_button_row("Reading/Playback", [READ_Q_CMDS[0], READ_A_CMDS[0], STOP_AUDIO_CMDS[0]], cols=3, disable_logic="not_taking", stop_audio=False) # Stop handled separately
        create_button_row("Answer Editing", [CLEAR_CMDS[0], UNDO_CMDS[0], DELETE_LAST_WORD_CMDS[0], BACKSPACE_CMDS[0]], cols=4, disable_logic="not_taking", stop_audio=True)
        create_button_row("Exam Control", [SAVE_PROGRESS_CMDS[0], SUBMIT_CMDS[0]], cols=2, disable_logic="not_taking", stop_audio=True)
        create_button_row("Submission Screen", [CONFIRM_SUBMIT_CMDS[0], CANCEL_SUBMIT_CMDS[0]], cols=2, disable_logic="not_submitting", stop_audio=True)

        # Special Inputs (Go To, Append)
        st.write("**Special Inputs:**")
        is_taking_exam = (current_app_state == VE_STATE_TAKING_EXAM)
        is_processing_voice = st.session_state.get('ve_processing_voice_command', False)
        special_inputs_disabled = not is_taking_exam or is_processing_voice

        cols_special = st.columns([1, 2])
        with cols_special[0]:
            max_q = len(st.session_state.ve_exam_data) if st.session_state.ve_exam_data else 1
            q_num_default = min(max_q, st.session_state.get('ve_current_q_index', 0) + 1)
            q_num = st.number_input("Go To #", min_value=1, max_value=max_q, value=q_num_default, step=1, key="ve_test_goto_num", label_visibility="collapsed", disabled=special_inputs_disabled)
        with cols_special[1]:
             if st.button(f"Go To Item {q_num}", key="ve_test_btn_goto", disabled=special_inputs_disabled, use_container_width=True):
                 log.debug(f"Voice Exam TEST UI: Button 'Go To Item {q_num}' clicked.")
                 stop_audio_playback()
                 st.session_state.ve_command_input_value = f"go to item {q_num}"
                 st.rerun()

        cols_append = st.columns([3, 1])
        with cols_append[0]:
            append_text = st.text_input("Append Text:", key="ve_test_append_text", label_visibility="collapsed", disabled=special_inputs_disabled)
        with cols_append[1]:
             if st.button("Append", key="ve_test_btn_append", disabled=(special_inputs_disabled or not append_text), use_container_width=True):
                  log.debug(f"Voice Exam TEST UI: Button 'Append' clicked: '{append_text}'")
                  st.session_state.ve_command_input_value = append_text
                  st.rerun()


def display_submission_review(config_dict):
    """Displays the final review screen."""
    st.header("Confirm Submission")
    st.warning("### Are you sure you want to submit your answers?")
    st.markdown("This action cannot be undone. Use sidebar commands or buttons below.")
    answered_count = len([a for a in st.session_state.get('ve_answers', {}).values() if a and str(a).strip()])
    total_questions = len(st.session_state.ve_exam_data) if st.session_state.ve_exam_data else 0
    unanswered_count = total_questions - answered_count
    st.info(f"Answers provided: **{answered_count}** / **{total_questions}** ({unanswered_count} unanswered)")

    if st.session_state.get('ve_answers'):
        with st.expander("Show Submitted Answers Preview"):
            exam_data = st.session_state.get('ve_exam_data', [])
            # Sort answers by question index for consistent display
            sorted_answers = sorted(st.session_state.ve_answers.items())
            for q_idx, answer in sorted_answers:
                 q_num = q_idx + 1
                 q_text_preview = "*Question text unavailable*"
                 if 0 <= q_idx < len(exam_data) and exam_data[q_idx]:
                     q_text_preview = exam_data[q_idx].get('original_text_chunk', '')[:50].strip() + "..."
                 st.markdown(f"**Q{q_num}:** *({q_text_preview})*")
                 st.text_area(
                     f"Answer {q_num}", value=answer if answer and str(answer).strip() else "<No answer provided>",
                     height=75, disabled=True, key=f"ve_review_{q_idx}"
                 )
                 st.markdown("---")

    confirm_cols = st.columns(2)
    with confirm_cols[0]:
        if st.button("✔️ YES, Submit Final", type="primary", use_container_width=True, key="ve_btn_confirm_submit"):
            log.debug("Voice Exam UI: Confirm Submit button clicked.")
            stop_audio_playback()
            st.session_state.ve_command_input_value = CONFIRM_SUBMIT_CMDS[0]
            st.rerun()
    with confirm_cols[1]:
        if st.button("↩️ NO, Return to Exam", type="secondary", use_container_width=True, key="ve_btn_cancel_submit"):
            log.debug("Voice Exam UI: Cancel Submit button clicked.")
            stop_audio_playback()
            st.session_state.ve_command_input_value = CANCEL_SUBMIT_CMDS[0]
            st.rerun()

    # Show test buttons even on submit screen for testing confirm/cancel via voice/text
    display_test_command_buttons(config_dict)

def display_finished_ui():
    """Displays the final screen after submission."""
    st.balloons()
    st.success("## Exam Submitted Successfully!")
    st.write("Your answers have been recorded.")
    final_feedback = st.session_state.get('ve_last_feedback', 'Submission complete.')
    st.info(final_feedback)
    if st.button("Start New Exam Session", key="ve_btn_start_another"):
        log.debug("Voice Exam UI: Start New Exam button clicked.")
        reset_to_selection_state()
        st.rerun()


# --- Main Rendering Logic ---

def render_voice_exam_page(config_dict):
    """Renders the voice exam page based on the current state."""
    log.debug("\n--- Voice Exam: render_voice_exam_page() - START ---")

    # Initialize state if it's the first time loading this feature
    if 've_app_state' not in st.session_state:
         initialize_voice_exam_state()
    # Validate paths on first load or if base path might have changed
    if 've_paths_validated' not in st.session_state:
         st.session_state.ve_paths_validated = validate_paths(config_dict)
    if not st.session_state.ve_paths_validated:
         st.error("Voice Exam feature cannot load due to invalid path configuration.")
         return # Stop rendering if paths are bad

    # Display Status Sidebar (always)
    display_status_sidebar(config_dict)

    app_state = st.session_state.ve_app_state
    log.debug(f"Voice Exam Main: Current state = {app_state}")
    log.debug(f"Voice Exam Main: Processing Voice Flag = {st.session_state.get('ve_processing_voice_command')}")

    # --- Timer Check ---
    start_time = st.session_state.get('ve_exam_start_time')
    if app_state == VE_STATE_TAKING_EXAM and start_time:
        duration_seconds = get_config_value(config_dict, 'VE_EXAM_DURATION_MINUTES', 60) * 60
        elapsed_time = time.time() - start_time
        remaining_time = duration_seconds - elapsed_time
        if remaining_time <= 0:
            log.info("Voice Exam Main: Time's up! Initiating auto-submit.")
            st.warning("Time's Up! Automatically submitting your exam...")
            save_successful = save_answers_to_file(config_dict, final_save=True)
            if not save_successful:
                 st.error("Auto-submit failed to save answers. Please try submitting manually.")
            log.debug("Voice Exam Main: Rerunning after auto-submit attempt.")
            st.rerun()
            return # Exit function after triggering rerun

    # --- Main Area Rendering based on State ---
    if app_state == VE_STATE_SELECTING_EXAM:
        log.debug(f"Voice Exam Main: Rendering UI for state {app_state} using display_selection_ui")
        display_selection_ui(config_dict)
    elif app_state == VE_STATE_TAKING_EXAM:
        log.debug(f"Voice Exam Main: Rendering UI for state {app_state} using display_exam_ui")
        # display_exam_ui is a helper function that combines elements
        item = get_current_exam_item()
        if item:
            with st.container():
                display_exam_item(item, st.session_state.ve_current_q_index, len(st.session_state.ve_exam_data))
                st.divider()
                display_answer_area()
                st.divider()
                display_pending_tts_audio() # Shows TTS player if path is set
                col1, col2 = st.columns(2)
                with col1:
                    display_navigation_buttons()
                with col2:
                    display_action_buttons(config_dict)
                display_test_command_buttons(config_dict) # Add test buttons
        else:
            st.error("Error: Could not display the current question. Exam data might be missing.")
            if st.button("Return to Exam Selection"):
                reset_to_selection_state()
                st.rerun()
    elif app_state == VE_STATE_SUBMITTING:
        log.debug(f"Voice Exam Main: Rendering UI for state {app_state} using display_submission_review")
        display_submission_review(config_dict)
    elif app_state == VE_STATE_FINISHED:
        log.debug(f"Voice Exam Main: Rendering UI for state {app_state} using display_finished_ui")
        display_finished_ui()
    else:
        log.error(f"Voice Exam ERROR Main: Unknown application state '{app_state}'. Resetting.")
        st.error(f"Error: Unknown application state '{app_state}'. Resetting to selection screen.")
        reset_to_selection_state()
        st.rerun()
        return

    # --- Process Submitted Command (from text, voice, or test button) ---
    # Check the specific session state variable for this feature
    submitted_command = st.session_state.get("ve_command_input_value", "")

    if submitted_command:
        log.debug(f"Voice Exam Main: Found submitted command: '{submitted_command}'")
        st.session_state.ve_command_input_value = "" # Clear BEFORE processing
        log.debug("Voice Exam Main: Cleared ve_command_input_value.")

        log.debug(f"Voice Exam Main: Calling process_text_input for '{submitted_command}'")
        needs_rerun = process_text_input(config_dict, submitted_command) # Pass config
        log.debug(f"Voice Exam Main: process_text_input returned needs_rerun = {needs_rerun}")

        # Reset the flag *after* the command has been processed
        if st.session_state.get('ve_processing_voice_command', False):
            log.debug("Voice Exam Main: Resetting ve_processing_voice_command flag to False.")
            st.session_state.ve_processing_voice_command = False

        if needs_rerun:
             log.debug(f"Voice Exam Main: Rerunning UI because needs_rerun is True.")
             st.rerun()
        else:
             log.debug(f"Voice Exam Main: No rerun triggered from main loop (needs_rerun is False).")
    else:
        # Reset flag if no command was processed but flag was somehow still true
        if st.session_state.get('ve_processing_voice_command', False):
            log.debug("Voice Exam Main: No command submitted, resetting lingering processing_voice_command flag.")
            st.session_state.ve_processing_voice_command = False
        # log.debug("Voice Exam Main: No submitted command found.")

    log.debug("--- Voice Exam: render_voice_exam_page() - END ---")

# --- END OF FILE voice_exam_helpers.py ---