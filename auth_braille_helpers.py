# --- START OF auth_braille_helpers.py ---

import streamlit as st
import os
import io
import time
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import logging
from pathlib import Path

# --- Logging Setup ---
log = logging.getLogger(__name__) # Use __name__ for logger hierarchy

# --- Feature-Specific Imports (Auth/Braille) ---
try:
    from deepface import DeepFace
except ImportError:
    log.warning("DeepFace library not found. Face auth features will be disabled.")
    DeepFace = None # Set to None if import fails

# --- Configuration (Auth, Braille) ---
# Base directory relative to this helper file (assuming it's in the main app dir)
_APP_DIR = Path(__file__).resolve().parent
APP1_BASE_DIR = _APP_DIR / "app1_data"
LETTER_AUDIO_DIR = APP1_BASE_DIR / "letters_wav"
REGISTERED_EMBEDDINGS_DIR = APP1_BASE_DIR / "registered_face_embeddings"
REGISTERED_VOICES_DIR = APP1_BASE_DIR / "registered_voices"
TEMP_AUDIO_DIR = APP1_BASE_DIR / "temp_audio" # Inside app1_data

# Bleep sound file names (paths constructed relative to APP1_BASE_DIR)
BEEP_HIGH_FILENAME = "beep_high.wav"
BEEP_MID_FILENAME  = "beep.wav"
BEEP_LOW_FILENAME  = "beep_low.wav"

# Audio Settings
BLEEP_DELAY_MS = 350
INTER_LETTER_DELAY_MS = 400
INITIAL_PAUSE_MS = 100
BLEEP_FADE_MS = 5
LEFT_PAN_PD = -0.9
RIGHT_PAN_PD = 0.9
SAMPLE_RATE = 16000
VOICE_REC_DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Authentication Settings
PREDEFINED_PHRASE = "My voice is my passport, verify me."
VOICE_SIMILARITY_THRESHOLD = 0.60
if DeepFace:
    FACE_MODEL_NAME = "Facenet512"
    FACE_SIMILARITY_THRESHOLD = 0.70
    FACE_VERIFICATION_DELAY = 2
else:
    FACE_MODEL_NAME = None
    FACE_SIMILARITY_THRESHOLD = None
    FACE_VERIFICATION_DELAY = None

# Braille Data
BRAILLE_PATTERNS = { # Standard Braille Patterns
    'A':(1,0,0,0,0,0),'B':(1,1,0,0,0,0),'C':(1,0,0,1,0,0),'D':(1,0,0,1,1,0),'E':(1,0,0,0,1,0),'F':(1,1,0,1,0,0),
    'G':(1,1,0,1,1,0),'H':(1,1,0,0,1,0),'I':(0,1,0,1,0,0),'J':(0,1,0,1,1,0),'K':(1,0,1,0,0,0),'L':(1,1,1,0,0,0),
    'M':(1,0,1,1,0,0),'N':(1,0,1,1,1,0),'O':(1,0,1,0,1,0),'P':(1,1,1,1,0,0),'Q':(1,1,1,1,1,0),'R':(1,1,1,0,1,0),
    'S':(0,1,1,1,0,0),'T':(0,1,1,1,1,0),'U':(1,0,1,0,0,1),'V':(1,1,1,0,0,1),'W':(0,1,0,1,1,1),'X':(1,0,1,1,0,1),
    'Y':(1,0,1,1,1,1),'Z':(1,0,1,0,1,1),
}
ALPHABET = sorted(BRAILLE_PATTERNS.keys())

# --- Ensure App1 Dirs Exist ---
# This code runs when the module is imported
try:
    APP1_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LETTER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    REGISTERED_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTERED_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Auth/Braille data directories ensured under: {APP1_BASE_DIR}")
except Exception as e:
    log.error(f"Failed to create mandatory Auth/Braille directories under {APP1_BASE_DIR}: {e}", exc_info=True)
    # We might want to raise an error here or let the app handle the failure downstream
    # For now, log the error. The functions using these dirs will likely fail later.


# --- Helper Functions (Auth/Braille) ---

@st.cache_resource # Cache loaded sounds
def load_bleep_sounds_st():
    """Loads the three pitched bleep sounds (.wav) using configured paths."""
    bleeps = {}
    paths = {
        "high": APP1_BASE_DIR / BEEP_HIGH_FILENAME,
        "mid":  APP1_BASE_DIR / BEEP_MID_FILENAME,
        "low":  APP1_BASE_DIR / BEEP_LOW_FILENAME,
    }
    loaded_correctly = True
    missing_files_msg = []
    # Use logging instead of Streamlit elements for background loading status
    log.info("Loading bleep sound resources...")

    for key, path_obj in paths.items():
        if not path_obj.exists():
            errmsg = f"Missing bleep WAV file: {path_obj.name} (Expected at: {path_obj})"
            log.error(errmsg)
            missing_files_msg.append(errmsg)
            loaded_correctly = False
            continue
        try:
            if path_obj.stat().st_size == 0:
                log.error(f"Bleep WAV file '{path_obj.name}' is empty at {path_obj}")
                loaded_correctly = False
                continue
            sound = AudioSegment.from_file(path_obj)
            bleeps[key] = sound.fade_in(BLEEP_FADE_MS).fade_out(BLEEP_FADE_MS)
            log.info(f"Successfully loaded {key} bleep from WAV: {path_obj}")
        except CouldntDecodeError:
            log.error(f"Error decoding {key} bleep WAV: {path_obj.name}. Is it a valid WAV file?")
            loaded_correctly = False
        except Exception as e:
            log.error(f"Error loading {key} bleep WAV ({path_obj.name}): {type(e).__name__} - {e}")
            loaded_correctly = False

    if loaded_correctly:
        log.info("All bleep sounds loaded successfully.")
        return bleeps
    else:
        # Log the failure but let the main app decide how to handle it (e.g., display st.error)
        log.error(f"Failed loading bleep WAVs. Errors:\n" + "\n".join(missing_files_msg))
        # Display error in Streamlit UI when this function is called if loading fails
        st.error("Failed to load one or more critical audio resources (bleeps). Braille audio disabled.")
        return None

@st.cache_resource
def load_registered_voices_st():
    """Loads registered voice embeddings from .npy files."""
    registered_data = {}
    if not REGISTERED_VOICES_DIR.exists():
        log.warning(f"Registered voices directory not found: {REGISTERED_VOICES_DIR}")
        return {}

    loaded_count = 0
    error_count = 0
    for filepath in REGISTERED_VOICES_DIR.glob("*.npy"):
        user_id = filepath.stem
        try:
            registered_data[user_id] = np.load(filepath)
            loaded_count += 1
        except Exception as e:
            st.warning(f"Error loading voice embedding for user '{user_id}': {e}")
            log.warning(f"Error loading voice file {filepath}: {e}", exc_info=True)
            error_count += 1

    log.info(f"Loaded {loaded_count} registered voice embeddings ({error_count} errors).")
    return registered_data

@st.cache_data # Cache the generated audio bytes based on letter and bleep sounds
def generate_braille_audio_bytes_st(letter, _bleep_sounds):
    """
    Generates combined Braille audio (letter announcement + dot pattern)
    using WAV files and exports as WAV bytes in memory.
    """
    if not isinstance(_bleep_sounds, dict) or not _bleep_sounds:
        st.error("Cannot generate Braille audio: Bleep sounds dictionary is invalid or empty.")
        log.error("generate_braille_audio_bytes_st called with invalid _bleep_sounds.")
        return None

    letter = letter.upper()
    if letter not in ALPHABET or letter not in BRAILLE_PATTERNS:
        st.error(f"Invalid letter '{letter}' for Braille audio generation.")
        log.error(f"Invalid letter '{letter}' passed to generate_braille_audio_bytes_st.")
        return None

    letter_file_path = LETTER_AUDIO_DIR / f"{letter}.wav"
    if not letter_file_path.is_file():
        log.error(f"Error: Letter WAV file not found: {letter_file_path}")
        st.error(f"Audio file for letter '{letter}' not found.")
        return None

    try:
        letter_segment = AudioSegment.from_file(str(letter_file_path))
    except Exception as e:
        log.error(f"Error loading letter WAV segment for '{letter}': {e}", exc_info=True)
        st.error(f"Error loading audio for letter '{letter}': {e}")
        return None

    final_audio = AudioSegment.silent(duration=INITIAL_PAUSE_MS) + letter_segment
    final_audio += AudioSegment.silent(duration=INTER_LETTER_DELAY_MS)

    pattern = BRAILLE_PATTERNS[letter]
    dot_sequence = [1, 2, 3, 4, 5, 6]
    bleep_sequence = AudioSegment.empty()

    for dot_number in dot_sequence:
        dot_index = dot_number - 1
        processed_bleep = AudioSegment.silent(duration=0)

        if pattern[dot_index] == 1:
            bleep_key = ""
            if dot_number in [1, 4]: bleep_key = "high"
            elif dot_number in [2, 5]: bleep_key = "mid"
            elif dot_number in [3, 6]: bleep_key = "low"

            if bleep_key and bleep_key in _bleep_sounds:
                base_bleep = _bleep_sounds[bleep_key]
                pan_val = LEFT_PAN_PD if dot_number <= 3 else RIGHT_PAN_PD
                processed_bleep = base_bleep.pan(pan_val)
            else:
                log.warning(f"Could not find loaded bleep sound for key '{bleep_key}' (dot {dot_number}, letter {letter}). Using silence.")

        bleep_sequence += processed_bleep
        padding_duration = max(0, BLEEP_DELAY_MS - len(processed_bleep))
        bleep_sequence += AudioSegment.silent(duration=padding_duration)

    final_audio += bleep_sequence

    try:
        buffer = io.BytesIO()
        final_audio.export(buffer, format="wav")
        log.debug(f"Successfully generated and exported combined WAV for letter {letter}")
        return buffer.getvalue()
    except Exception as e:
        log.error(f"Error exporting combined WAV audio for letter {letter}: {e}", exc_info=True)
        st.error(f"Error exporting generated audio: {e}")
        return None

@st.cache_resource # Cache loaded face embeddings
def load_registered_embeddings_st():
    """Loads registered face embeddings from .npy files."""
    known_face_embeddings = []
    known_face_names = []
    if DeepFace is None:
        log.warning("DeepFace not available, cannot load face embeddings.")
        return [], []

    if not REGISTERED_EMBEDDINGS_DIR.exists():
        log.warning(f"Registered face embeddings directory not found: {REGISTERED_EMBEDDINGS_DIR}")
        return [], []

    loaded_count = 0
    error_count = 0
    for filepath in REGISTERED_EMBEDDINGS_DIR.glob("*.npy"):
        user_id = filepath.stem
        try:
            embedding = np.load(filepath).flatten()
            known_face_embeddings.append(embedding)
            known_face_names.append(user_id)
            loaded_count += 1
        except Exception as e:
            st.warning(f"Error loading face embedding for user '{user_id}': {e}")
            log.warning(f"Error loading face embedding file {filepath}: {e}", exc_info=True)
            error_count += 1

    log.info(f"Loaded {loaded_count} registered face embeddings ({error_count} errors).")
    return known_face_embeddings, known_face_names

def run_face_verification_window(known_embeddings, known_names):
    """
    Opens a camera window using OpenCV for real-time face verification.
    Uses constants defined in this module (FACE_MODEL_NAME, etc.)
    Returns: tuple: (success (bool), user_id (str | None))
    """
    if DeepFace is None:
        st.error("Face verification requires the DeepFace library, which is not available.")
        log.error("run_face_verification_window called but DeepFace is None.")
        return False, None
    if not known_embeddings or not known_names:
        st.warning("No face embeddings registered for verification.")
        log.warning("run_face_verification_window called with no registered embeddings.")
        return False, None

    st.info("Starting face verification... Please look at the camera.")
    time.sleep(1) # User prep time

    video_capture = None
    success = False
    verified_user_id = None
    window_name = 'Face Verification - Press Q to Quit'

    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            log.warning("Camera index 0 failed, trying index 1...")
            video_capture = cv2.VideoCapture(1)
            if not video_capture.isOpened():
                st.error("Could not open webcam. Check connection/permissions.")
                log.error("Failed to open camera index 0 and 1.")
                return False, None

        cv2.namedWindow(window_name)
        last_verification_time = 0
        tentative_match = False
        current_tentative_user = None

        while True:
            ret, frame = video_capture.read()
            if not ret:
                log.error("Failed to capture frame from webcam.")
                st.error("Error capturing video frame.")
                break

            display_frame = frame.copy()
            detected_embedding, facial_area = None, None

            try:
                # Note: Uses FACE_MODEL_NAME constant from this module
                objs = DeepFace.represent(
                    img_path=frame,
                    model_name=FACE_MODEL_NAME,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if objs and isinstance(objs, list) and len(objs) > 0:
                     detected_embedding = np.array(objs[0]["embedding"], dtype='float32')
                     facial_area = objs[0]["facial_area"]
            except Exception as represent_err:
                 log.warning(f"DeepFace represent error: {represent_err}", exc_info=False)

            matched_name = "Unknown"
            max_similarity = -1.0

            if detected_embedding is not None and facial_area is not None:
                similarities = cosine_similarity(
                    detected_embedding.reshape(1, -1),
                    np.array(known_embeddings)
                )
                best_match_index = np.argmax(similarities[0])
                max_similarity = similarities[0][best_match_index]

                # Note: Uses FACE_SIMILARITY_THRESHOLD constant
                if max_similarity >= FACE_SIMILARITY_THRESHOLD:
                    matched_name = known_names[best_match_index]

                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                label = f"{matched_name} ({max_similarity:.2f})"
                color = (0, 0, 255) # Red default

                if matched_name != "Unknown":
                    if not tentative_match or current_tentative_user != matched_name:
                        tentative_match = True
                        current_tentative_user = matched_name
                        last_verification_time = time.time()
                        color = (255, 255, 0) # Cyan tentative
                        label = f"Verifying: {label}"
                    elif current_tentative_user == matched_name:
                        # Note: Uses FACE_VERIFICATION_DELAY constant
                        if time.time() - last_verification_time > FACE_VERIFICATION_DELAY:
                            success = True
                            verified_user_id = matched_name
                            color = (0, 255, 0) # Green success
                            label = f"Verified: {matched_name}"
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.imshow(window_name, display_frame)
                            cv2.waitKey(1500)
                            break
                        else:
                            color = (255, 255, 0) # Cyan tentative
                            label = f"Verifying: {label}"

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            else:
                cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tentative_match = False
                current_tentative_user = None

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("Face verification cancelled by user (Q pressed).")
                break

    except Exception as e:
        st.error(f"An error occurred during face verification: {e}")
        log.error(f"Face verification window error: {e}", exc_info=True)
        success = False

    finally:
        if video_capture and video_capture.isOpened():
            video_capture.release()
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1) # Ensure window closes fully

    if success:
        # Let the main app display success message
        log.info(f"Face verification successful for user: {verified_user_id}")
        return True, verified_user_id
    else:
        log.info(f"Face verification failed or cancelled. Last tentative user: {current_tentative_user}")
        # Let the main app display failure message
        return False, None

def record_audio_st(filename: str, duration: int = VOICE_REC_DURATION, sr: int = SAMPLE_RATE):
    """
    Records audio from the default microphone using constants from this module.
    Displays UI feedback using Streamlit.
    Returns: str | None: The filename if successful, None otherwise.
    """
    # Use Streamlit elements for user feedback during recording
    st.info(f"Please prepare to say the phrase: '{PREDEFINED_PHRASE}'")
    time.sleep(1.0)

    placeholder = st.empty()
    try:
        with placeholder.container():
            st.write("Recording starts in:")
            st.write("3...")
            time.sleep(1)
            st.write("2...")
            time.sleep(1)
            st.write("1...")
            time.sleep(1)
            st.write("🔴 RECORDING NOW...")

        # Start recording using constants from this module
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        placeholder.success("✅ Recording finished.")
        log.info(f"Audio recording finished ({duration}s).")

        file_path = Path(filename)
        # Ensure parent directory exists (uses TEMP_AUDIO_DIR implicitly via caller)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(file_path, recording, sr)
        log.info(f"Audio recording saved to: {file_path}")
        return filename

    except sd.PortAudioError as pa_err:
         log.error(f"PortAudio error during recording: {pa_err}", exc_info=True)
         st.error(f"Audio Recording Error: Could not access microphone. Details: {pa_err}")
         placeholder.error("Audio Recording Error!")
         return None
    except Exception as e:
        log.error(f"Unexpected error during audio recording: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during recording: {e}")
        placeholder.error("Audio Recording Error!")
        return None

def extract_mel_spectrogram_st(audio_path: str, sr: int = SAMPLE_RATE, n_mels: int = N_MELS, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
    """
    Extracts log Mel spectrogram using constants from this module.
    Returns: np.ndarray | None: The processed spectrogram, or None on error.
    """
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        if sr_loaded != sr:
            log.warning(f"Audio loaded with sample rate {sr_loaded}, resampling to {sr}.")

        y_trimmed, index = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) == 0:
             log.warning(f"Audio signal empty after trimming: {audio_path}. Skipping.")
             st.warning("Audio signal is silent or too quiet after trimming.")
             return None
        log.debug(f"Audio trimmed from {len(y)/sr:.2f}s to {len(y_trimmed)/sr:.2f}s.")

        mel_spec = librosa.feature.melspectrogram(
            y=y_trimmed, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad/truncate using VOICE_REC_DURATION constant
        target_len_frames = int(np.ceil(VOICE_REC_DURATION * sr / hop_length))
        current_len_frames = log_mel_spec.shape[1]

        if current_len_frames < target_len_frames:
            pad_width = target_len_frames - current_len_frames
            log_mel_spec_padded = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='minimum')
            log.debug(f"Padded spectrogram from {current_len_frames} to {target_len_frames} frames.")
            return log_mel_spec_padded
        elif current_len_frames > target_len_frames:
            log_mel_spec_truncated = log_mel_spec[:, :target_len_frames]
            log.debug(f"Truncated spectrogram from {current_len_frames} to {target_len_frames} frames.")
            return log_mel_spec_truncated
        else:
            log.debug(f"Spectrogram already at target length: {target_len_frames} frames.")
            return log_mel_spec

    except FileNotFoundError:
         st.error(f"Audio file not found: {audio_path}")
         log.error(f"Audio file not found for spectrogram extraction: {audio_path}")
         return None
    except Exception as e:
        st.error(f"Error processing audio for Mel spectrogram: {e}")
        log.error(f"Error extracting Mel spectrogram from {audio_path}: {e}", exc_info=True)
        return None

# --- END OF auth_braille_helpers.py ---