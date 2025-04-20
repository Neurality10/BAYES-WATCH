# text_to_speech.py
"""
Handles Text-to-Speech generation by saving to a local directory.
Stores the file path in session state for the main UI to display the player.
"""

import streamlit as st
from gtts import gTTS, gTTSError
import os
import traceback
import requests
import io
import time

from config import get_config_value

# Define a directory to store the generated TTS files
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_OUTPUT_DIR = os.path.join(_BASE_DIR, "tts_output")

# Ensure the output directory exists
try:
    os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
    print(f"INFO TTS: Ensured TTS output directory exists: {TTS_OUTPUT_DIR}")
except OSError as e:
    print(f"ERROR TTS: Could not create TTS output directory '{TTS_OUTPUT_DIR}': {e}")
    st.error(f"Could not create TTS output directory: {e}. TTS saving will likely fail.")


# Cache the audio generation *to bytes* (optional, kept for consistency)
@st.cache_data(show_spinner=False)
def get_tts_audio_bytes(text, lang='en-US'):
    """Generates TTS audio bytes using gTTS. Returns bytes or None."""
    # ... (Implementation remains the same) ...
    operation = f"get_tts_audio_bytes(text='{text[:50]}...', lang='{lang}')"
    print(f"DEBUG TTS BYTES - ENTER: {operation}")
    if not text or not isinstance(text, str) or not text.strip():
        print(f"DEBUG TTS BYTES - SKIP: Empty or invalid text provided.")
        return None
    try:
        tts_lang = lang if lang else get_config_value('TTS_LANGUAGE', 'en-US')
        tts_lang_short = tts_lang.split('-')[0]
        print(f"DEBUG TTS BYTES - INFO: Using language code '{tts_lang_short}' for gTTS.")
        tts = gTTS(text=text, lang=tts_lang_short, slow=False)
        fp = io.BytesIO()
        print(f"DEBUG TTS BYTES - ACTION: Calling gTTS.write_to_fp() for '{text[:50]}...'")
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        print(f"DEBUG TTS BYTES - RESULT: gTTS.write_to_fp() finished. Bytes generated: {len(audio_bytes)}")
        if not audio_bytes:
            error_msg = f"TTS Error: gTTS returned 0 bytes for text '{text[:30]}...' (lang: {tts_lang_short})."
            print(f"ERROR TTS BYTES: {error_msg}")
            st.session_state.last_feedback = error_msg
            return None
        print(f"DEBUG TTS BYTES - SUCCESS: Generation successful for lang '{tts_lang_short}'.")
        print(f"DEBUG TTS BYTES - EXIT: {operation} - Returning {len(audio_bytes)} bytes.")
        return audio_bytes
    except gTTSError as ge:
         error_msg = f"TTS Generation Error: {ge}. Lang='{tts_lang_short}'. Check internet/API?"
         print(f"ERROR TTS BYTES: {error_msg}")
         st.session_state.last_feedback = error_msg
         traceback.print_exc()
         print(f"DEBUG TTS BYTES - EXIT: {operation} - Returning None due to gTTSError.")
         return None
    except requests.exceptions.RequestException as ce:
         error_msg = f"TTS Network Error: Cannot connect ({type(ce).__name__}). Check internet. Details: {ce}"
         print(f"ERROR TTS BYTES: {error_msg}")
         st.session_state.last_feedback = error_msg
         traceback.print_exc()
         print(f"DEBUG TTS BYTES - EXIT: {operation} - Returning None due to RequestException.")
         return None
    except Exception as e:
        error_msg = f"Unexpected Error generating TTS: {type(e).__name__} - {e}"
        print(f"ERROR TTS BYTES: {error_msg}")
        st.session_state.last_feedback = error_msg
        traceback.print_exc()
        print(f"DEBUG TTS BYTES - EXIT: {operation} - Returning None due to unexpected Exception.")
        return None

def trigger_tts_playback(text):
    """
    Generates TTS, saves it to a LOCAL directory (tts_output/), and stores the
    file path in st.session_state['tts_file_to_play'] for the UI to display.
    Files are NOT deleted automatically.
    Updates status/feedback but DOES NOT call st.rerun().
    Returns True if TTS file was successfully generated and path stored, False otherwise.
    """
    operation = f"trigger_tts_playback(text='{text[:50]}...')"
    print(f"DEBUG TTS TRIGGER - ENTER: {operation}")

    # Clear any previous TTS file path first
    if 'tts_file_to_play' in st.session_state:
        del st.session_state['tts_file_to_play']
        print("DEBUG TTS TRIGGER - Cleared previous tts_file_to_play path.")


    # --- Update Status ---
    st.session_state.audio_status = "Processing TTS..."
    st.session_state.last_feedback = f"Generating speech for: '{text[:30]}...'"
    print(f"DEBUG TTS TRIGGER - STATE: Set audio_status='Processing TTS...', feedback='{st.session_state.last_feedback}'")

    if not text or not isinstance(text, str) or not text.strip():
        print(f"DEBUG TTS TRIGGER - SKIP: Empty or invalid text provided.")
        st.session_state.audio_status = "Error: No text"
        st.session_state.last_feedback = "Cannot generate speech for empty text."
        return False

    # --- Generation and Saving to Local File ---
    output_file_path = None
    success = False # Flag to track if audio file generated ok
    try:
        # Select language
        tts_lang = get_config_value('TTS_LANGUAGE', 'en-US')
        tts_lang_short = tts_lang.split('-')[0]
        print(f"DEBUG TTS TRIGGER - INFO: Using language code '{tts_lang_short}' for gTTS.")

        # 1. Determine file path in the local output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text_prefix = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
        if not safe_text_prefix: safe_text_prefix = "tts"
        filename = f"{timestamp}_{safe_text_prefix}.mp3"
        output_file_path = os.path.join(TTS_OUTPUT_DIR, filename)
        print(f"DEBUG TTS TRIGGER - ACTION: Determined output file path: {output_file_path}")

        # 2. Create gTTS object and save to the path
        tts = gTTS(text=text, lang=tts_lang_short, slow=False)
        print(f"DEBUG TTS TRIGGER - ACTION: Calling gTTS.save() to '{output_file_path}'")
        tts.save(output_file_path)
        print(f"DEBUG TTS TRIGGER - SUCCESS: gTTS.save() finished.")

        # 3. Check if file has content
        if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
             error_msg = f"TTS Error: gTTS failed to save or saved an empty file for text '{text[:30]}...' (lang: {tts_lang_short}). Path: {output_file_path}"
             print(f"ERROR TTS TRIGGER: {error_msg}")
             st.session_state.last_feedback = error_msg
             st.session_state.audio_status = "Error: Save failed/Empty"
             return False

        # 4. Store the path in session state for the UI to pick up
        st.session_state['tts_file_to_play'] = output_file_path
        print(f"DEBUG TTS TRIGGER - STATE: Stored TTS path in session_state: {output_file_path}")

        # *** UPDATED FEEDBACK ***
        st.session_state.last_feedback = f"🔊 TTS ready: '{text[:30]}...'. Player loading below.\n(Saved as: {os.path.basename(output_file_path)})"

        # Update status
        st.session_state.audio_status = "Ready (TTS)" # Indicate it's ready to be played
        print(f"DEBUG TTS TRIGGER - STATE: Set audio_status='Ready (TTS)', feedback='{st.session_state.last_feedback}'")
        success = True # Mark that the file was generated

    # --- Error Handling ---
    except gTTSError as ge:
         error_msg = f"TTS Generation/Save Error: {ge}. Lang='{tts_lang_short}'. Check internet/API?"
         print(f"ERROR TTS TRIGGER: {error_msg}")
         st.session_state.last_feedback = error_msg
         st.session_state.audio_status = "Error: Generation"
         traceback.print_exc()
    except requests.exceptions.RequestException as ce:
         error_msg = f"TTS Network Error: Cannot connect ({type(ce).__name__}). Check internet. Details: {ce}"
         print(f"ERROR TTS TRIGGER: {error_msg}")
         st.session_state.last_feedback = error_msg
         st.session_state.audio_status = "Error: Network"
         traceback.print_exc()
    except OSError as oe:
         error_msg = f"TTS File System Error: {oe}. Check path/permissions for '{TTS_OUTPUT_DIR}'?"
         print(f"ERROR TTS TRIGGER: {error_msg}")
         st.session_state.last_feedback = error_msg
         st.session_state.audio_status = "Error: File System"
         traceback.print_exc()
    except Exception as e:
        error_msg = f"Unexpected Error during TTS processing: {type(e).__name__} - {e}"
        print(f"ERROR TTS TRIGGER: {error_msg}")
        st.session_state.last_feedback = error_msg
        st.session_state.audio_status = "Error: Processing"
        traceback.print_exc()

    print(f"DEBUG TTS TRIGGER - EXIT: {operation} - Returning {success}.")
    return success


def stop_audio_playback():
    """
    Clears any pending TTS file path from session state and sets status to Idle.
    """
    operation = "stop_audio_playback()"
    print(f"DEBUG AUDIO - ENTER: {operation}")
    status_changed = False

    # Clear the pending TTS file path
    if 'tts_file_to_play' in st.session_state:
        print(f"DEBUG AUDIO - ACTION: Clearing pending TTS file path: {st.session_state['tts_file_to_play']}")
        del st.session_state['tts_file_to_play']
        status_changed = True # Clearing the path is a change

    # Update status if not Idle
    current_status = st.session_state.get("audio_status", "Idle")
    if current_status != "Idle":
        print(f"DEBUG AUDIO - STATE: Changing status from '{current_status}' to 'Idle'.")
        st.session_state.audio_status = "Idle"
        st.session_state.last_feedback = "Stopped audio playback / Cleared pending TTS."
        status_changed = True
    else:
        # If status was already idle, but we might have cleared a path, update feedback
        if status_changed:
             st.session_state.last_feedback = "Cleared pending TTS."
        print(f"DEBUG AUDIO - STATE: Status already 'Idle', no status change needed (Path cleared: {status_changed}).")


    print(f"DEBUG AUDIO - EXIT: {operation} (Status changed: {status_changed})")