# ui_elements.py
"""Functions to render specific UI components for the text-input based exam app."""

import streamlit as st
import time
import os
import glob # Needed for finding files within exam run dir
from config import get_config_value
from state_manager import (STATE_TAKING_EXAM, STATE_SUBMITTING, STATE_FINISHED,
                           STATE_SELECTING_EXAM, reset_to_selection_state)
from data_handler import (get_current_exam_item, get_current_audio_path,
                          save_current_answer, save_answers_to_file, load_exam_data,
                          find_available_exams) # Import exam finder
from text_to_speech import stop_audio_playback, trigger_tts_playback

# Import STT components
from streamlit_mic_recorder import mic_recorder
from speech_to_text import transcribe_audio, get_recognizer

# Import command keywords (or fallback)
try:
    from command_processor import (NAV_CMDS, PREV_CMDS, READ_Q_CMDS, READ_A_CMDS,
                                   CLEAR_CMDS, UNDO_CMDS, DELETE_LAST_WORD_CMDS,
                                   BACKSPACE_CMDS, SUBMIT_CMDS, CONFIRM_SUBMIT_CMDS,
                                   CANCEL_SUBMIT_CMDS, SAVE_PROGRESS_CMDS, STOP_AUDIO_CMDS)
    print("DEBUG UI: Imported command lists from command_processor.")
except ImportError:
    print("WARN UI: Could not import command lists. Using fallback definitions.")
    NAV_CMDS = ["next"]
    PREV_CMDS = ["previous"]
    READ_Q_CMDS = ["read question"]
    READ_A_CMDS = ["read answer"]
    CLEAR_CMDS = ["clear"]
    UNDO_CMDS = ["undo"]
    DELETE_LAST_WORD_CMDS = ["delete last word"]
    BACKSPACE_CMDS = ["backspace"]
    SUBMIT_CMDS = ["submit"]
    CONFIRM_SUBMIT_CMDS = ["confirm"]
    CANCEL_SUBMIT_CMDS = ["cancel"]
    SAVE_PROGRESS_CMDS = ["save"]
    STOP_AUDIO_CMDS = ["stop"]


# --- Main UI Display Functions ---

def display_status_sidebar():
    """Displays status indicators, feedback, and command input in the sidebar."""
    with st.sidebar:
        st.header("Status")

        # --- Timer ---
        start_time = st.session_state.get('exam_start_time')
        current_app_state = st.session_state.get('app_state', STATE_SELECTING_EXAM)
        if start_time and current_app_state not in [STATE_FINISHED, STATE_SELECTING_EXAM]:
            duration_seconds = get_config_value('EXAM_DURATION_MINUTES', 60) * 60
            elapsed_time = time.time() - start_time
            remaining_time = duration_seconds - elapsed_time
            if remaining_time < 0: remaining_time = 0
            timer_text = "Time's Up!" if remaining_time <= 0 else f"⏳ {int(remaining_time // 60):02d}:{int(remaining_time % 60):02d}"
            is_urgent = (0 < remaining_time < 300) or (remaining_time <= 0)
            color = "red" if is_urgent else "inherit"
            st.markdown(f"<p style='font-size: 1.2em; color: {color};'>Time Left: {timer_text}</p>", unsafe_allow_html=True)
        elif current_app_state == STATE_FINISHED:
             st.markdown("<p style='font-size: 1.2em;'>Time Left: Exam Finished</p>", unsafe_allow_html=True)
        else:
             st.markdown("<p style='font-size: 1.2em;'>Time Left: --:--</p>", unsafe_allow_html=True)

        st.divider()

        # --- Speaker Status (TTS) ---
        audio_status = st.session_state.get("audio_status", "Idle")
        if "Ready" in audio_status: audio_icon = "▶️"
        elif "Processing" in audio_status: audio_icon = "⏳"
        elif "Error" in audio_status: audio_icon = "⚠️"
        else: audio_icon = "✔️"
        st.metric("TTS Status", f"{audio_status} {audio_icon}")

        st.divider()

        # --- Feedback Area ---
        st.subheader("Feedback")
        feedback = st.session_state.get('last_feedback', "Initializing...")
        st.info(feedback)

        st.divider()

        # --- Inputs only shown when Taking/Submitting ---
        input_disabled = current_app_state not in [STATE_TAKING_EXAM, STATE_SUBMITTING]
        if not input_disabled:
            # --- Command Input Area (Text) ---
            st.subheader("Command / Dictate (Text)")
            command_key = f"command_input_{st.session_state.get('command_input_key', 0)}"
            user_input = st.text_input(
                "Type command OR text to append, then press Enter:",
                key=command_key,
                placeholder="e.g., next, read question, clear...",
                label_visibility="collapsed",
                disabled=input_disabled,
                value=""
            )
            if user_input:
                if not st.session_state.get('processing_voice_command', False):
                    print(f"DEBUG UI: Text input submitted: '{user_input}'")
                    st.session_state.command_input_value = user_input
                else:
                    print("DEBUG UI: Text input ignored while processing voice command.")

            # --- Voice Input Area (STT) ---
            st.subheader("Command / Dictate (Voice)")
            stt_generally_disabled = input_disabled
            recognizer = get_recognizer()
            if recognizer is None or st.session_state.get('speech_rec_error'):
                error_detail = st.session_state.get('last_feedback', "Unknown initialization error.")
                if "Voice recognizer init error:" in error_detail:
                    error_detail = error_detail.split("Voice recognizer init error:", 1)[1].strip()
                st.warning(f"Speech recognizer not available. Voice input disabled.\nReason: {error_detail}", icon="🎙️")
                stt_specifically_disabled = True
            else:
                stt_specifically_disabled = False
            stt_disabled = stt_generally_disabled or stt_specifically_disabled

            mic_key = f"mic_recorder_{st.session_state.get('command_input_key', 0)}"
            audio_info = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                key=mic_key,
                use_container_width=True,
                format="wav"
            )
            if audio_info and not stt_disabled and not st.session_state.get('processing_voice_command', False):
                print(f"DEBUG UI: Mic recorder returned audio data (bytes: {len(audio_info.get('bytes', 0))}). STT enabled & not processing, proceeding.")
                st.session_state.last_feedback = "Audio recorded, attempting transcription..."
                transcribed_text = transcribe_audio(audio_info, recognizer)
                if transcribed_text:
                    print(f"DEBUG UI: Transcription successful: '{transcribed_text}'. Setting command_input_value and processing flag.")
                    st.session_state.processing_voice_command = True
                    st.session_state.command_input_value = transcribed_text
                    st.session_state.command_input_key += 1
                    print(f"DEBUG UI: Incremented command_input_key to {st.session_state.command_input_key} before STT rerun.")
                    st.rerun()
                else:
                    print("DEBUG UI: Transcription failed or returned no text.")
            elif audio_info and stt_disabled:
                print(f"DEBUG UI: Mic recorder returned audio data, but STT is disabled. Ignoring.")
            elif audio_info and st.session_state.get('processing_voice_command', False):
                print("DEBUG UI: Mic recorder returned audio data, but already processing previous voice command. Ignoring.")

            # --- Command List Expander ---
            with st.expander("Show Available Commands"):
                st.markdown("""
                 *   Use the **Text Input** OR the **Voice Recorder** above.
                 *   Recognized commands (`next`, `clear`, etc.) are executed.
                 *   Other spoken/typed text is appended to the current answer.

                 **Navigation:** `next`, `previous`, `go to item <number>`
                 **Reading/Playback:** `read question`, `read answer`, `stop`
                 **Answer Editing:** `(Type/Speak text)`, `clear`, `delete last word`, `backspace`, `undo`
                 **Exam Control:** `save`, `submit`
                 **Submission:** `confirm`, `cancel`
                 """)
            if current_app_state == STATE_SUBMITTING:
                st.caption("Submit confirmation active: Use `confirm` or `cancel`")
        else:
             st.caption("Controls disabled until exam starts or resumes.")


# --- Exam Selection UI ---
def display_selection_ui():
    """Displays the initial exam selection/start screen."""
    st.title("Text & Voice Examination")
    st.markdown("Welcome! Please select an exam run below.")

    if not st.session_state.get('available_exams'):
         st.session_state.available_exams = find_available_exams()
    available_exams = st.session_state.available_exams
    exam_names = list(available_exams.keys())

    if not exam_names:
         st.error(f"No valid exam runs found in the configured directory: `{get_config_value('PIPELINE_OUTPUT_BASE_DIR')}`. Please check the configuration and the directory structure.")
         st.info(f"Expected structure: `<base_dir>/{get_config_value('EXAM_RUN_DIR_PREFIX')}*`/`<exam_id>`/`<exam_id>_transcripts.json` and `<base_dir>/{get_config_value('EXAM_RUN_DIR_PREFIX')}*`/`<exam_id>`/`audio_gtts`")
         st.stop()

    selected_display_name = st.selectbox(
        "Select Exam:",
        options=exam_names,
        key="exam_selector",
        index=exam_names.index(st.session_state.selected_exam_name) if st.session_state.selected_exam_name in exam_names else 0
    )
    st.session_state.selected_exam_name = selected_display_name

    if st.button(f"Load Exam: {selected_display_name}", type="primary", key="load_exam_button"):
        print(f"DEBUG UI: Load button clicked for exam: {selected_display_name}")
        selected_run_path = available_exams.get(selected_display_name)
        if not selected_run_path:
            st.error("Internal error: Could not find path for selected exam name.")
            return

        st.session_state.selected_exam_run_path = selected_run_path
        st.session_state.exam_name = selected_display_name

        json_pattern = get_config_value("EXAM_JSON_GLOB_PATTERN")
        audio_pattern = get_config_value("EXAM_AUDIO_DIR_GLOB_PATTERN")
        json_search_path = os.path.join(selected_run_path, json_pattern)
        audio_search_path = os.path.join(selected_run_path, audio_pattern)
        found_json_files = glob.glob(json_search_path)
        found_audio_dirs = [d for d in glob.glob(audio_search_path) if os.path.isdir(d)]

        if not found_json_files:
            st.error(f"Could not find '*_transcripts.json' file within the selected exam run folder structure: {selected_run_path}/{json_pattern}")
            return
        if not found_audio_dirs:
             st.error(f"Could not find 'audio_gtts' directory within the selected exam run folder structure: {selected_run_path}/{audio_pattern}")
             return

        st.session_state.current_exam_json_path = found_json_files[0]
        st.session_state.current_audio_base_path = found_audio_dirs[0]
        print(f"DEBUG UI: Set JSON path: {st.session_state.current_exam_json_path}")
        print(f"DEBUG UI: Set Audio path: {st.session_state.current_audio_base_path}")

        recognizer_init = get_recognizer()
        if recognizer_init is None:
             st.error("Failed to initialize voice recognizer. Voice input will be unavailable.")

        if load_exam_data():
            st.session_state.app_state = STATE_TAKING_EXAM
            st.session_state.exam_start_time = time.time()
            st.session_state.current_q_index = 0
            st.session_state.answers = {}
            st.session_state.undo_stack = {}
            if 'tts_file_to_play' in st.session_state:
                del st.session_state['tts_file_to_play']
            st.session_state.last_feedback = f"Exam '{selected_display_name}' started! Type or speak commands/text in the sidebar."
            st.session_state.current_answer_key = 1
            st.session_state.command_input_key += 1
            st.session_state.command_input_value = ""
            st.session_state.processing_voice_command = False
            print(f"INFO UI: Exam '{selected_display_name}' loaded. Transitioning to STATE_TAKING_EXAM.")
            st.rerun()
        else:
             st.error("Failed to load exam data for the selected run. Please check file integrity.")

    st.markdown("---")
    st.subheader("Instructions")
    st.info("""
    *   Select the desired exam run from the dropdown above and click "Load Exam".
    *   Once loaded, the exam interface will appear.
    *   **Listen (Question):** Click the ▶️ button below the question number to listen to pre-recorded audio, if available.
    *   **Listen (TTS):** Use `read question` or `read answer` commands in the sidebar. An audio player will appear temporarily in the main page area. Click its ▶️ button to play.
    *   **Answer:** Type your full answer directly into the **'Your Answer'** box in the main area. Changes are saved automatically.
    *   **Use Sidebar for Commands & Dictation:**
        *   **Type:** Use the **Text Input** box for commands or to append text.
        *   **Speak:** Use the **🎤 Start Recording** button for commands or to append text via voice.
    *   **Submit:** Use the `submit` command or the 'Finish Exam' button when done.
    """)


# --- Exam Taking UI components ---

def display_exam_item(item, index, total_items):
    """Displays the current question text and the st.audio player using file path."""
    if not item:
        st.warning("No exam item data found for this question.")
        return
    st.subheader(f"Question {index + 1} of {total_items}")
    audio_path = get_current_audio_path()
    if audio_path:
        try:
            print(f"DEBUG UI: Displaying st.audio player for Q{index + 1} using path: {audio_path}")
            # *** FIX: Removed invalid key argument ***
            st.audio(audio_path, format="audio/mpeg")
        except FileNotFoundError:
             print(f"ERROR UI: Audio file not found by st.audio: {audio_path}")
             st.caption("(Audio file missing. Use 'read question' for TTS.)")
        except Exception as e:
             print(f"ERROR UI: Failed to display st.audio for {audio_path}: {e}")
             st.caption(f"(Error loading audio: {e}. Use 'read question' for TTS.)")
    else:
        print(f"DEBUG UI: No pre-recorded audio path found or file missing for Q{index + 1}")
        st.caption("(No pre-recorded audio available. Use 'read question' for TTS.)")
    question_text = item.get('original_text_chunk', '*No question text found.*')
    st.markdown(f"> {question_text}")


def display_answer_area():
    """Displays the text area for the current answer."""
    current_index = st.session_state.current_q_index
    answer_key = f"answer_area_{current_index}_{st.session_state.current_answer_key}"
    current_answer_text = st.session_state.answers.get(current_index, "")
    label = "Your Answer (Changes saved automatically):"
    def _save_typed_answer():
        if answer_key in st.session_state:
            new_value = st.session_state[answer_key]
            if st.session_state.answers.get(current_index, "") != new_value:
                save_current_answer(new_value)
        else:
             print(f"WARN UI: Answer area key '{answer_key}' not found in session state during save callback.")
    st.text_area(
        label, value=current_answer_text, height=300, key=answer_key,
        placeholder="Type your full answer here. Use the sidebar to append text or issue editing commands like 'clear', 'undo', 'delete last word'.",
        on_change=_save_typed_answer, label_visibility="visible"
    )


def display_pending_tts_audio():
    """Checks session state for a TTS file path and displays st.audio if found."""
    tts_file = st.session_state.get('tts_file_to_play')
    if tts_file and os.path.exists(tts_file):
        print(f"DEBUG UI: Found pending TTS file: {tts_file}. Displaying in main area.")
        try:
            with st.container():
                st.write("---")
                st.subheader("Text-to-Speech Output")
                # *** FIX: Removed invalid key argument ***
                st.audio(tts_file, format='audio/mp3')
                st.caption(f"(Playing TTS: {os.path.basename(tts_file)})")
                st.write("---")
        except FileNotFoundError:
            print(f"ERROR UI: Pending TTS file disappeared before rendering? Path: {tts_file}")
            st.warning("The generated TTS audio file could not be found.")
        except Exception as e:
            print(f"ERROR UI: Failed to display pending TTS audio {tts_file}: {e}")
            st.error(f"Error loading TTS audio player: {e}")
        # Clear path AFTER attempting display
        del st.session_state['tts_file_to_play']
        print("DEBUG UI: Cleared pending TTS file path after attempting display.")
    elif tts_file: # Path was set, but file doesn't exist
        print(f"WARN UI: Pending TTS file path found ({tts_file}), but file does not exist.")
        st.warning("Could not find the generated TTS audio file.")
        del st.session_state['tts_file_to_play']


def display_navigation_buttons():
    """Displays Previous/Next buttons (part of normal UI)."""
    current_index = st.session_state.current_q_index
    total_questions = len(st.session_state.exam_data) if st.session_state.exam_data else 0
    nav_cols = st.columns(2)
    with nav_cols[0]:
        prev_disabled = (current_index == 0)
        if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True, key="btn_prev"):
             print("DEBUG UI: Previous button clicked.")
             stop_audio_playback()
             st.session_state.command_input_value = PREV_CMDS[0]
             st.rerun()
    with nav_cols[1]:
        next_disabled = (current_index >= total_questions - 1)
        if st.button("Next ➡️", disabled=next_disabled, use_container_width=True, key="btn_next"):
            print("DEBUG UI: Next button clicked.")
            stop_audio_playback()
            st.session_state.command_input_value = NAV_CMDS[0]
            st.rerun()


def display_action_buttons():
    """Displays Save Progress, Finish buttons (part of normal UI)."""
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("💾 Save Progress", use_container_width=True, key="btn_save"):
             print("DEBUG UI: Save Progress button clicked.")
             stop_audio_playback()
             st.session_state.command_input_value = SAVE_PROGRESS_CMDS[0]
             st.rerun()
    with action_cols[1]:
        if st.button("🏁 Finish Exam", type="secondary", use_container_width=True, key="btn_finish"):
             print("DEBUG UI: Finish Exam button clicked.")
             stop_audio_playback()
             st.session_state.command_input_value = SUBMIT_CMDS[0]
             st.rerun()


# --- Test Command Button Function ---
def display_test_command_buttons():
    """Displays buttons for all commands in an expander for testing."""
    st.write("---")
    with st.expander("🧪 TESTING: Trigger Commands Manually", expanded=False):
        current_app_state = st.session_state.get('app_state', STATE_SELECTING_EXAM)
        print(f"DEBUG UI TEST: Current state for test buttons = {current_app_state}")

        def create_button_row(label, commands, cols=3, disable_logic=None, stop_audio=True):
            st.write(f"**{label}:**")
            columns = st.columns(cols)
            col_idx = 0
            for cmd in commands:
                with columns[col_idx % cols]:
                    disabled = False
                    if disable_logic == "not_taking":
                        disabled = (current_app_state != STATE_TAKING_EXAM)
                    elif disable_logic == "not_submitting":
                         disabled = (current_app_state != STATE_SUBMITTING)
                    elif disable_logic == "always_enabled":
                        disabled = False
                    if cmd in CONFIRM_SUBMIT_CMDS or cmd in CANCEL_SUBMIT_CMDS:
                         if st.session_state.get('processing_voice_command', False): disabled = True
                    if cmd in SUBMIT_CMDS or cmd in SAVE_PROGRESS_CMDS:
                         if st.session_state.get('processing_voice_command', False): disabled = True

                    if st.button(cmd.title(), key=f"test_btn_{cmd.replace(' ', '_')}", disabled=disabled, use_container_width=True):
                        print(f"DEBUG UI TEST: Button '{cmd}' clicked.")
                        if stop_audio:
                             stop_audio_playback()
                        st.session_state.command_input_value = cmd
                        st.rerun()
                col_idx += 1
            st.write("---")

        create_button_row("Navigation", [NAV_CMDS[0], PREV_CMDS[0]], cols=2, disable_logic="not_taking")
        create_button_row("Reading/Playback", [READ_Q_CMDS[0], READ_A_CMDS[0], STOP_AUDIO_CMDS[0]], cols=3, disable_logic=None, stop_audio=False)
        create_button_row("Answer Editing", [CLEAR_CMDS[0], UNDO_CMDS[0], DELETE_LAST_WORD_CMDS[0], BACKSPACE_CMDS[0]], cols=4, disable_logic="not_taking")
        create_button_row("Exam Control", [SAVE_PROGRESS_CMDS[0], SUBMIT_CMDS[0]], cols=2, disable_logic="not_taking")
        create_button_row("Submission Screen", [CONFIRM_SUBMIT_CMDS[0], CANCEL_SUBMIT_CMDS[0]], cols=2, disable_logic="not_submitting")

        st.write("**Special Inputs:**")
        cols_special = st.columns([1, 2])
        with cols_special[0]:
            max_q = len(st.session_state.exam_data) if st.session_state.exam_data else 1
            q_num_default = min(max_q, st.session_state.get('current_q_index', 0) + 1)
            q_num = st.number_input("Go To #", min_value=1, max_value=max_q, value=q_num_default, step=1, key="test_goto_num", label_visibility="collapsed", disabled=(current_app_state != STATE_TAKING_EXAM))
        with cols_special[1]:
             if st.button(f"Go To Item {q_num}", key="test_btn_goto", disabled=(current_app_state != STATE_TAKING_EXAM), use_container_width=True):
                 print(f"DEBUG UI TEST: Button 'Go To Item {q_num}' clicked.")
                 stop_audio_playback()
                 st.session_state.command_input_value = f"go to item {q_num}"
                 st.rerun()

        cols_append = st.columns([3, 1])
        with cols_append[0]:
            append_text = st.text_input("Append Text:", key="test_append_text", label_visibility="collapsed", disabled=(current_app_state != STATE_TAKING_EXAM))
        with cols_append[1]:
             if st.button("Append", key="test_btn_append", disabled=(current_app_state != STATE_TAKING_EXAM or not append_text), use_container_width=True):
                  print(f"DEBUG UI TEST: Button 'Append' clicked with text: '{append_text}'")
                  st.session_state.command_input_value = append_text
                  st.rerun()


# --- Submission/Finished UI ---
def display_submission_review():
    """Displays the final review screen before final submission."""
    st.header("Confirm Submission")
    st.warning("### Are you sure you want to submit your answers?")
    st.markdown("This action cannot be undone. Review summary below.")
    st.markdown("Use sidebar commands `confirm` or `cancel`, or the buttons below (or test buttons in expander).")
    answered_count = len([a for a in st.session_state.get('answers', {}).values() if a and str(a).strip()])
    total_questions = len(st.session_state.exam_data) if st.session_state.exam_data else 0
    unanswered_count = total_questions - answered_count
    st.info(f"Answers provided for **{answered_count}** / **{total_questions}** questions. ({unanswered_count} unanswered)")
    if st.session_state.get('answers'):
        with st.expander("Show Submitted Answers Preview"):
            exam_data = st.session_state.get('exam_data', [])
            sorted_answers = sorted(st.session_state.answers.items())
            for q_idx, answer in sorted_answers:
                 q_num = q_idx + 1
                 q_text_preview = "*Question text unavailable*"
                 if 0 <= q_idx < len(exam_data) and exam_data[q_idx]:
                     q_text_preview = exam_data[q_idx].get('original_text_chunk', '')[:50].strip() + "..."
                 st.markdown(f"**Q{q_num}:** *({q_text_preview})*")
                 st.text_area(
                     f"Answer {q_num}",
                     value=answer if answer and str(answer).strip() else "<No answer provided>",
                     height=75, disabled=True, key=f"review_{q_idx}"
                 )
                 st.markdown("---", unsafe_allow_html=True)
    confirm_cols = st.columns(2)
    with confirm_cols[0]:
        if st.button("✔️ YES, Submit Final", type="primary", use_container_width=True, key="btn_confirm_submit"):
            print("DEBUG UI: Confirm Submit button clicked.")
            stop_audio_playback()
            st.session_state.command_input_value = CONFIRM_SUBMIT_CMDS[0]
            st.rerun()
    with confirm_cols[1]:
        if st.button("↩️ NO, Return to Exam", type="secondary", use_container_width=True, key="btn_cancel_submit"):
            print("DEBUG UI: Cancel Submit button clicked.")
            stop_audio_playback()
            st.session_state.command_input_value = CANCEL_SUBMIT_CMDS[0]
            st.rerun()
    display_test_command_buttons()


def display_finished_ui():
    """Displays the final screen after successful submission."""
    st.balloons()
    st.success("## Exam Submitted Successfully!")
    st.write("Your answers have been recorded.")
    final_feedback = st.session_state.get('last_feedback', 'Submission complete.')
    st.info(final_feedback)
    if st.button("Start New Exam Session", key="btn_start_another"):
        print("DEBUG UI: Start New Exam button clicked.")
        reset_to_selection_state()
        st.rerun()