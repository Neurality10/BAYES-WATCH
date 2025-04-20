# command_processor.py
"""Processes text commands entered in the sidebar."""

import streamlit as st
import re
from state_manager import STATE_TAKING_EXAM, STATE_SUBMITTING, STATE_FINISHED
from data_handler import (get_current_exam_item, save_current_answer,
                          add_undo_state, save_answers_to_file)
# *** CHANGE: Import the new trigger function ***
from text_to_speech import trigger_tts_playback, stop_audio_playback

# --- Command Keywords (Unchanged) ---
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

def process_text_input(text_input):
    """
    Parses the text input and executes the corresponding action.
    Returns True if a rerun is needed after processing, False otherwise.
    TTS commands now trigger generation and store path; rerun needed to display player.
    """
    operation = f"process_text_input('{text_input[:50]}...')"
    print(f"DEBUG CMD - ENTER: {operation}")

    if not text_input or not isinstance(text_input, str):
        print(f"DEBUG CMD - SKIP: Invalid input.")
        return False

    command = text_input.strip().lower()
    original_text = text_input.strip()

    rerun_needed = True # Default: Assume UI needs refresh

    # --- Stop Audio / Clear Pending TTS Command (Check First) ---
    if command in STOP_AUDIO_CMDS:
        print(f"DEBUG CMD - MATCH: STOP_AUDIO_CMDS ('{command}')")
        stop_audio_playback() # Clears pending TTS path and updates status
        print(f"DEBUG CMD - EXIT: {operation} after stop_audio_playback. Returning True.")
        return True # Rerun needed to show updated status/feedback

    # --- State-Specific Commands ---
    current_state = st.session_state.app_state
    if current_state == STATE_SUBMITTING:
        print(f"DEBUG CMD - STATE: Processing command '{command}' in STATE_SUBMITTING")
        # Stop audio is good practice before confirm/cancel actions
        stop_audio_playback()

        if command in CONFIRM_SUBMIT_CMDS:
            print("DEBUG CMD - MATCH: CONFIRM_SUBMIT_CMDS")
            st.session_state.last_feedback = "Submitting final answers..."
            if not save_answers_to_file(final_save=True):
                st.session_state.last_feedback = "Error saving submission. Please try again or cancel."
            rerun_needed = True
        elif command in CANCEL_SUBMIT_CMDS:
            print("DEBUG CMD - MATCH: CANCEL_SUBMIT_CMDS")
            st.session_state.app_state = STATE_TAKING_EXAM
            st.session_state.last_feedback = "Submission cancelled. Returned to exam."
            rerun_needed = True
        else:
            print(f"DEBUG CMD - INVALID: Command '{command}' invalid during submission.")
            feedback_msg = f"'{original_text}' is not valid during confirmation. Use 'confirm' or 'cancel'."
            st.session_state.last_feedback = feedback_msg
            # Maybe trigger TTS feedback here?
            trigger_tts_playback(feedback_msg)
            rerun_needed = True

        print(f"DEBUG CMD - EXIT: {operation} from STATE_SUBMITTING. Returning {rerun_needed}.")
        return rerun_needed

    # --- General Commands (STATE_TAKING_EXAM) ---
    elif current_state == STATE_TAKING_EXAM:
        print(f"DEBUG CMD - STATE: Processing command '{command}' in STATE_TAKING_EXAM")
        # Stop audio *before* non-reading commands.
        if command not in READ_Q_CMDS and command not in READ_A_CMDS:
             print("DEBUG CMD - ACTION: Stopping pending TTS for non-reading command.")
             stop_audio_playback() # Clears pending path, updates status

        current_q_index = st.session_state.current_q_index
        total_questions = len(st.session_state.exam_data) if st.session_state.exam_data else 0
        current_answer = st.session_state.answers.get(current_q_index, "")

        # --- Navigation ---
        if command in NAV_CMDS:
            print("DEBUG CMD - MATCH: NAV_CMDS")
            stop_audio_playback() # Ensure pending TTS is cleared on nav
            if current_q_index < total_questions - 1:
                st.session_state.current_q_index += 1
                st.session_state.current_answer_key += 1
                st.session_state.last_feedback = f"Moved to Question {st.session_state.current_q_index + 1}"
                rerun_needed = True
            else:
                st.session_state.last_feedback = "Already at the last question."
                rerun_needed = False
        elif command in PREV_CMDS:
            print("DEBUG CMD - MATCH: PREV_CMDS")
            stop_audio_playback() # Ensure pending TTS is cleared on nav
            if current_q_index > 0:
                st.session_state.current_q_index -= 1
                st.session_state.current_answer_key += 1
                st.session_state.last_feedback = f"Moved to Question {st.session_state.current_q_index + 1}"
                rerun_needed = True
            else:
                st.session_state.last_feedback = "Already at the first question."
                rerun_needed = False

        # --- TTS Reading ---
        elif command in READ_Q_CMDS:
            print("DEBUG CMD - MATCH: READ_Q_CMDS")
            stop_audio_playback() # Clear previous pending TTS first
            item = get_current_exam_item()
            text_to_read = item.get('original_text_chunk', '') if item else ''
            print(f"DEBUG CMD - INFO: Text to read for question: '{text_to_read[:50]}...'")
            if text_to_read:
                trigger_tts_playback(text_to_read) # Stores path, updates status/feedback
            else:
                feedback_msg = "No question text found to read."
                st.session_state.last_feedback = feedback_msg
                trigger_tts_playback(feedback_msg) # Read the feedback
            rerun_needed = True # Rerun needed to display player / show feedback update

        elif command in READ_A_CMDS:
            print("DEBUG CMD - MATCH: READ_A_CMDS")
            stop_audio_playback() # Clear previous pending TTS first
            current_answer_to_read = st.session_state.answers.get(current_q_index, "").strip()
            print(f"DEBUG CMD - INFO: Text to read for answer: '{current_answer_to_read[:50]}...'")
            if current_answer_to_read:
                trigger_tts_playback(current_answer_to_read) # Stores path, updates status/feedback
            else:
                feedback_msg = "No answer recorded yet to read."
                st.session_state.last_feedback = feedback_msg
                trigger_tts_playback(feedback_msg) # Read the feedback itself
            rerun_needed = True # Rerun needed to display player / show feedback update

        # --- Answer Manipulation ---
        elif command in CLEAR_CMDS:
            print("DEBUG CMD - MATCH: CLEAR_CMDS")
            if current_answer:
                add_undo_state(current_q_index, current_answer)
                st.session_state.answers[current_q_index] = ""
                st.session_state.last_feedback = "Answer cleared."
                st.session_state.current_answer_key += 1
                rerun_needed = True
            else:
                st.session_state.last_feedback = "Answer is already empty."
                rerun_needed = False
        elif command in UNDO_CMDS:
            print("DEBUG CMD - MATCH: UNDO_CMDS")
            if current_q_index in st.session_state.undo_stack and st.session_state.undo_stack[current_q_index]:
                restored_answer = st.session_state.undo_stack[current_q_index].pop()
                st.session_state.answers[current_q_index] = restored_answer
                st.session_state.last_feedback = "Undo successful."
                st.session_state.current_answer_key += 1
                rerun_needed = True
            else:
                st.session_state.last_feedback = "Nothing to undo for this question."
                rerun_needed = False
        elif command in DELETE_LAST_WORD_CMDS:
            print("DEBUG CMD - MATCH: DELETE_LAST_WORD_CMDS")
            if current_answer:
                add_undo_state(current_q_index, current_answer)
                words = current_answer.strip().split()
                if words:
                    words.pop()
                    new_answer = ' '.join(words)
                    if new_answer: new_answer += ' '
                    st.session_state.answers[current_q_index] = new_answer
                    st.session_state.last_feedback = "Deleted last word."
                else:
                    st.session_state.answers[current_q_index] = ""
                    st.session_state.last_feedback = "Answer cleared (contained only whitespace)."
                st.session_state.current_answer_key += 1
                rerun_needed = True
            else:
                st.session_state.last_feedback = "Answer is empty, cannot delete last word."
                rerun_needed = False
        elif command in BACKSPACE_CMDS:
            print("DEBUG CMD - MATCH: BACKSPACE_CMDS")
            if current_answer:
                add_undo_state(current_q_index, current_answer)
                st.session_state.answers[current_q_index] = current_answer[:-1]
                st.session_state.last_feedback = "Backspace applied."
                st.session_state.current_answer_key += 1
                rerun_needed = True
            else:
                st.session_state.last_feedback = "Answer is empty, cannot backspace."
                rerun_needed = False

        # --- Submission / Saving ---
        elif command in SUBMIT_CMDS:
            print("DEBUG CMD - MATCH: SUBMIT_CMDS")
            stop_audio_playback() # Clear pending TTS before submitting
            st.session_state.app_state = STATE_SUBMITTING
            st.session_state.last_feedback = "Please confirm submission."
            rerun_needed = True
        elif command in SAVE_PROGRESS_CMDS:
            print("DEBUG CMD - MATCH: SAVE_PROGRESS_CMDS")
            stop_audio_playback() # Clear pending TTS before saving
            save_answers_to_file(final_save=False) # Feedback handled internally
            rerun_needed = True

        # --- Go To Item Command ---
        elif (match := re.match(r"go to (?:item|question|number)\s+(\d+)", command)):
            print("DEBUG CMD - MATCH: GO_TO_ITEM")
            stop_audio_playback() # Clear pending TTS on nav
            num_str = match.group(1)
            try:
                num = int(num_str)
                target_idx = num - 1
                if 0 <= target_idx < total_questions:
                    if current_q_index != target_idx:
                        st.session_state.current_q_index = target_idx
                        st.session_state.current_answer_key += 1
                        st.session_state.last_feedback = f"Moved to Question {num}."
                        rerun_needed = True
                    else:
                        st.session_state.last_feedback = f"Already at Question {num}."
                        rerun_needed = False
                else:
                    st.session_state.last_feedback = f"Invalid question number: {num}. Max is {total_questions}."
                    trigger_tts_playback(st.session_state.last_feedback) # Read feedback
                    rerun_needed = True
            except ValueError:
                st.session_state.last_feedback = f"Could not understand question number: '{num_str}'."
                trigger_tts_playback(st.session_state.last_feedback) # Read feedback
                rerun_needed = True

        # --- Default: Append Text ---
        else:
            print(f"DEBUG CMD - DEFAULT: Append text '{original_text[:50]}...'")
            # No need to stop audio here, appending is passive
            add_undo_state(current_q_index, current_answer)
            separator = ' ' if current_answer and not current_answer.endswith(' ') else ''
            new_answer = current_answer + separator + original_text
            st.session_state.answers[current_q_index] = new_answer
            st.session_state.last_feedback = f"Appended: \"{original_text[:30]}...\""
            st.session_state.current_answer_key += 1
            rerun_needed = True

        print(f"DEBUG CMD - EXIT: {operation} from STATE_TAKING_EXAM. Returning {rerun_needed}.")
        return rerun_needed

    # --- Handle commands if not in TAKING_EXAM or SUBMITTING ---
    else:
        print(f"DEBUG CMD - STATE: Command '{command}' ignored in state {current_state}.")
        st.session_state.last_feedback = f"Command '{original_text}' ignored in current state ({current_state})."
        # Allow stopping audio even in finished state (already handled by initial check)
        rerun_needed = True # Rerun to show feedback msg

        print(f"DEBUG CMD - EXIT: {operation} from state {current_state}. Returning {rerun_needed}.")
        return rerun_needed