# state_manager.py
"""Defines application states and initializes session state for text input version."""

import streamlit as st
import time

# --- Application States ---
STATE_SELECTING_EXAM = "SELECTING_EXAM"
STATE_TAKING_EXAM = "TAKING_EXAM"
STATE_SUBMITTING = "SUBMITTING" # Confirmation screen state
STATE_FINISHED = "FINISHED"   # Final state after successful submission

# States where the main exam UI (question, answer, nav buttons) should be shown
EXAM_UI_STATES = {STATE_TAKING_EXAM}

def initialize_session_state():
    """Sets up the default session state variables if they don't exist."""
    defaults = {
        'app_state': STATE_SELECTING_EXAM,
        'exam_data': None,           # Loaded exam questions/items
        'exam_name': None,           # Name of the currently loaded exam
        'current_q_index': 0,
        'answers': {},               # Dict: {q_index: answer_text}
        'undo_stack': {},            # Dict: {q_index: [previous_answers]}
        'exam_start_time': None,
        'audio_status': "Idle",      # Status of TTS playback
        'last_feedback': "Initializing...", # Text feedback in sidebar
        'current_answer_key': 0,     # Used to force remount text_area
        'command_input_key': 0,      # Used to force clear text_input/mic
        'command_input_value': "",   # Stores the submitted command/text
        'tts_file_to_play': None,    # Stores path for pending TTS playback
        'speech_rec_error': False,   # Flag if recognizer init failed
        'processing_voice_command': False, # Flag for STT processing loop

        # --- NEW/MODIFIED for Exam Selection ---
        'available_exams': {},       # Dict: {display_name: run_folder_path}
        'selected_exam_name': None,  # User-friendly name selected by user
        'selected_exam_run_path': None, # Full path to the selected exam run folder
        'current_exam_json_path': None, # Dynamically set path to the JSON file
        'current_audio_base_path': None, # Dynamically set path to the audio folder
    }
    initialized_now = False
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            initialized_now = True

    if st.session_state.get('app_state') is None:
         print("WARNING State: app_state was None, resetting to SELECTING_EXAM")
         st.session_state.app_state = STATE_SELECTING_EXAM
         initialized_now = True

    if initialized_now:
         print("DEBUG State: Initialized/reset some session state variables.")


def reset_to_selection_state(clear_processed_list=True):
    """Resets state variables to start over from exam selection."""
    print("DEBUG State: Resetting state to selection...")

    # Keep track of keys to potentially delete (or reset)
    keys_to_reset = [
        'app_state', 'exam_data', 'exam_name', 'current_q_index', 'answers',
        'undo_stack', 'exam_start_time', 'audio_status', 'last_feedback',
        'current_answer_key', 'command_input_key', 'command_input_value',
        'tts_file_to_play', 'selected_exam_name', 'speech_rec_error',
        'processing_voice_command',
        'selected_exam_run_path', 'current_exam_json_path', 'current_audio_base_path'
        # Keep 'available_exams' maybe? Or recalculate? Let's recalculate for freshness.
    ]
    if 'available_exams' in st.session_state:
         del st.session_state['available_exams']

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize with defaults
    initialize_session_state()
    print("INFO: Session state reset to selection complete.")