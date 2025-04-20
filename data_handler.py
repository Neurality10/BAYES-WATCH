# data_handler.py
"""Handles finding exams, loading exam data, and managing answers."""

import streamlit as st
import json
import os
import glob # For finding files/folders using patterns
import traceback
import time
from config import get_config_value
from state_manager import STATE_FINISHED

# --- Exam Discovery ---

def find_available_exams():
    """
    Scans the configured PIPELINE_OUTPUT_BASE_DIR for valid exam run folders.
    Returns a dictionary mapping user-friendly exam names to their run folder paths.
    """
    base_path = get_config_value("PIPELINE_OUTPUT_BASE_DIR")
    run_prefix = get_config_value("EXAM_RUN_DIR_PREFIX", "streamlit_run_")
    json_pattern = get_config_value("EXAM_JSON_GLOB_PATTERN", os.path.join("*", "*_transcripts.json"))
    audio_pattern = get_config_value("EXAM_AUDIO_DIR_GLOB_PATTERN", os.path.join("*", "audio_gtts"))

    available_exams = {}
    print(f"DEBUG EXAM FIND: Scanning base path '{base_path}' for runs starting with '{run_prefix}'")

    if not base_path or not os.path.isdir(base_path):
        st.error(f"Exam base directory not found or configured: {base_path}")
        return {}

    try:
        for item_name in os.listdir(base_path):
            item_path = os.path.join(base_path, item_name)
            # Check if it's a directory and starts with the prefix
            if os.path.isdir(item_path) and item_name.startswith(run_prefix):
                print(f"DEBUG EXAM FIND: Found potential run folder: {item_name}")
                # Attempt to extract a user-friendly name
                display_name = item_name.replace(run_prefix, "", 1) # Remove prefix once

                # Verify required contents exist within this run folder using glob patterns
                json_search_path = os.path.join(item_path, json_pattern)
                audio_search_path = os.path.join(item_path, audio_pattern)

                found_json_files = glob.glob(json_search_path)
                found_audio_dirs = [d for d in glob.glob(audio_search_path) if os.path.isdir(d)]

                if found_json_files and found_audio_dirs:
                    # For simplicity, assume the first match is the correct one if multiple found
                    if len(found_json_files) > 1:
                        print(f"WARN EXAM FIND: Multiple JSON files found matching pattern in {item_path}. Using first: {found_json_files[0]}")
                    if len(found_audio_dirs) > 1:
                        print(f"WARN EXAM FIND: Multiple audio directories found matching pattern in {item_path}. Using first: {found_audio_dirs[0]}")

                    print(f"DEBUG EXAM FIND: Valid exam run found: '{display_name}' at path '{item_path}'")
                    available_exams[display_name] = item_path
                else:
                    if not found_json_files:
                         print(f"DEBUG EXAM FIND: Skipping '{item_name}', no JSON found matching '{json_search_path}'")
                    if not found_audio_dirs:
                         print(f"DEBUG EXAM FIND: Skipping '{item_name}', no audio dir found matching '{audio_search_path}'")

    except Exception as e:
        st.error(f"Error scanning for exams in '{base_path}': {e}")
        print(f"ERROR EXAM FIND: Exception during scan: {e}")
        traceback.print_exc()
        return {}

    print(f"DEBUG EXAM FIND: Found {len(available_exams)} valid exams.")
    return available_exams


# --- Exam Data Loading ---
def load_exam_data():
    """
    Loads the exam data from the JSON file specified in
    st.session_state.current_exam_json_path.
    """
    # *** Get path from session state ***
    filepath = st.session_state.get("current_exam_json_path")

    if not filepath:
         st.error("Error: Exam JSON path not set in session state.")
         print("ERROR DataLoad: current_exam_json_path not found in session state.")
         st.session_state.exam_data = None
         st.session_state.last_feedback = "Error: Could not determine which exam file to load."
         return False

    st.session_state.last_feedback = f"Loading exam data from {os.path.basename(filepath)}..."
    print(f"DEBUG DataLoad: Loading exam data from SESSION STATE path: {filepath}...")

    try:
        if not os.path.isfile(filepath):
            print(f"ERROR DataLoad: File not found at path: '{filepath}'")
            st.session_state.exam_data = None
            st.session_state.last_feedback = f"Error: Exam file not found: {os.path.basename(filepath)}"
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Rest of validation logic remains the same...
        if not isinstance(data, list):
             print(f"ERROR DataLoad: Expected a list in JSON file '{filepath}', got {type(data)}.")
             st.session_state.exam_data = None
             st.session_state.last_feedback = "Error: Invalid exam file format (must be a list)."
             return False

        valid_data = []
        required_keys = ['id', 'filename_base', 'original_text_chunk']
        for i, item in enumerate(data):
             if isinstance(item, dict) and all(key in item for key in required_keys):
                 item['internal_q_index'] = i
                 valid_data.append(item)
             else:
                 print(f"WARN DataLoad: Skipping invalid item at index {i} in {filepath}. Missing keys or not a dict.")

        if not valid_data:
             print(f"ERROR DataLoad: No valid exam items found in '{filepath}'.")
             st.session_state.exam_data = None
             st.session_state.last_feedback = "Error: No valid questions found in the exam file."
             return False

        try:
            valid_data.sort(key=lambda x: int(x.get('id', x['internal_q_index'])))
            print(f"DEBUG DataLoad: Sorted exam data by 'id' field.")
        except (ValueError, TypeError) as sort_err:
            print(f"WARN DataLoad: Could not sort by 'id' ({sort_err}), keeping internal index order.")
            valid_data.sort(key=lambda x: x['internal_q_index'])

        st.session_state.exam_data = valid_data
        # Exam name is now set during selection, but we can update it here if needed
        # st.session_state.exam_name = os.path.basename(filepath).replace('_transcripts.json', '')
        st.session_state.last_feedback = f"Exam '{st.session_state.exam_name}' loaded ({len(valid_data)} questions)."
        print(f"DEBUG DataLoad: Loaded {len(valid_data)} items for exam '{st.session_state.exam_name}'.")
        return True

    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON from '{os.path.basename(filepath)}'. Check format. Details: {e}")
        st.session_state.last_feedback = "Error: Invalid JSON format in exam file."
        print(f"ERROR DataLoad: JSON Decode Error in '{filepath}': {e}")
        st.session_state.exam_data = None
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred loading exam data: {e}")
        st.session_state.last_feedback = f"Error loading exam data: {e}"
        print(f"ERROR DataLoad: Unexpected error loading data from '{filepath}': {e}")
        traceback.print_exc()
        st.session_state.exam_data = None
        return False

def get_current_exam_item():
    """Returns the data for the current question index."""
    if not st.session_state.get('exam_data'):
        print("DEBUG Data: get_current_exam_item called with no exam_data.")
        return None
    index = st.session_state.get('current_q_index', 0)
    if 0 <= index < len(st.session_state.exam_data):
        return st.session_state.exam_data[index]
    print(f"WARN Data: Invalid current_q_index: {index} (Total items: {len(st.session_state.exam_data)})")
    return None

# --- Audio Handling (File Path for st.audio) ---
def _get_audio_path_for_item(item):
    """Internal helper to get the full audio path for a given item dictionary."""
    if not item or 'filename_base' not in item:
        print("WARN AudioPath: Item missing or no 'filename_base' key.")
        return None

    # *** Get base path from session state ***
    audio_base_path = st.session_state.get("current_audio_base_path")
    if not audio_base_path:
        print(f"ERROR AudioPath: current_audio_base_path not set in session state.")
        return None
    if not os.path.isdir(audio_base_path):
         print(f"ERROR AudioPath: Audio base path in session state is not a valid directory: {audio_base_path}")
         return None

    filename_base = item['filename_base']
    audio_filename = filename_base + ".mp3" # Assume MP3
    audio_path = os.path.join(audio_base_path, audio_filename)
    return audio_path

def get_current_audio_path():
    """Gets the file path for the current question's pre-recorded audio."""
    item = get_current_exam_item()
    if not item:
        print("DEBUG AudioPath: No current exam item found for audio.")
        return None
    audio_path = _get_audio_path_for_item(item)
    if not audio_path:
         print("DEBUG AudioPath: Could not determine audio path for current item.")
         return None

    if os.path.isfile(audio_path):
        return audio_path
    else:
        print(f"WARN AudioPath: Audio file not found at expected path: {audio_path}")
        return None


# --- Answer Management (remains the same) ---
def save_current_answer(answer_text):
    """
    Updates the answer in session state for the current question index.
    Also adds the *previous* state to the undo stack.
    """
    current_index = st.session_state.get('current_q_index', -1)
    if current_index < 0 or not isinstance(st.session_state.get('answers'), dict):
         print("WARN AnswerSave: Invalid index or answers state. Cannot save.")
         return
    previous_answer = st.session_state.answers.get(current_index, "")
    current_answer_str = str(answer_text) if answer_text is not None else ""
    if current_answer_str != previous_answer:
        add_undo_state(current_index, previous_answer)
        st.session_state.answers[current_index] = current_answer_str

def add_undo_state(index, previous_answer_state):
    """Adds a previous answer state to the undo stack for a given index."""
    if not isinstance(st.session_state.get('undo_stack'), dict):
        st.session_state.undo_stack = {}
        print("WARN Undo: Undo stack was not initialized correctly.")
    if index not in st.session_state.undo_stack:
        st.session_state.undo_stack[index] = []
    if not st.session_state.undo_stack[index] or st.session_state.undo_stack[index][-1] != previous_answer_state:
        max_undo_depth = 10
        if len(st.session_state.undo_stack[index]) >= max_undo_depth:
            st.session_state.undo_stack[index].pop(0)
        st.session_state.undo_stack[index].append(previous_answer_state)

def save_answers_to_file(final_save=False):
    """Saves the collected answers to a JSON file in the submissions directory."""
    print(f"INFO Data: Attempting save. Final save: {final_save}")
    current_answers = st.session_state.get('answers')
    if not current_answers:
         st.warning("No answers recorded yet.")
         st.session_state.last_feedback = "No answers to save."
         return False

    exam_data = st.session_state.get('exam_data', [])
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
             print(f"WARN SaveAnswers: Index {q_idx} out of bounds for exam_data (len {len(exam_data)}).")
         item_info = {
             "question_index": q_idx + 1,
             "answer_text": answer if answer is not None else "",
             **original_item_info
         }
         submission_answers.append(item_info)

    submission_answers.sort(key=lambda x: x["question_index"])

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Use the currently loaded exam name for the submission filename
        exam_name_slug = st.session_state.get('exam_name', 'exam').replace(' ', '_').replace('.', '')
        submissions_dir = get_config_value("SUBMISSIONS_DIR", "submissions")
        os.makedirs(submissions_dir, exist_ok=True)
        filename = os.path.join(submissions_dir, f"{exam_name_slug}_submission_{timestamp}.json")

        try:
            display_path = os.path.relpath(filename, start=os.getcwd())
        except ValueError:
            display_path = filename

        submission_data = {
            "exam_name": st.session_state.get('exam_name', 'Unknown'),
            "submission_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exam_start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.session_state.get('exam_start_time'))) if st.session_state.get('exam_start_time') else None,
            "total_questions_in_exam": len(exam_data),
            "questions_answered": len([a for a in current_answers.values() if a and str(a).strip()]),
            "answers": submission_answers
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, indent=2, ensure_ascii=False)

        feedback = f"Progress saved to `{display_path}`"
        if final_save:
            feedback = f"Exam submitted successfully! Saved to `{display_path}`"
            st.session_state.app_state = STATE_FINISHED
            st.session_state.last_feedback = feedback
            print(f"INFO Data: Final submission saved to {filename}. Transitioning to FINISHED.")
            return True
        else:
            st.session_state.last_feedback = feedback
            st.success(feedback)
            print(f"INFO Data: Progress saved to {filename}.")
            return True

    except Exception as e:
         feedback = f"Error saving submission file: {e}"
         st.session_state.last_feedback = feedback
         print(f"ERROR Data: Failed to save submission: {e}")
         traceback.print_exc()
         st.error(feedback)
         return False