import streamlit as st
import os
import io
import time
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import cv2 # For face verification window trigger
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
# CouldntDecodeError might still happen if WAV is corrupted, keep it
from pydub.exceptions import CouldntDecodeError
from gtts import gTTS

# --- Page Config (MUST BE FIRST Streamlit command) ---
st.set_page_config(page_title="Accessible Braille Learner (WAV)", layout="wide")

# --- Configuration ---
# !! ADJUST ALL PATHS !!
BASE_DIR = r"C:\Users\HARSH\Desktop\hack36" # Base directory
# !! Point to the directory with LETTER .wav files !!
LETTER_AUDIO_DIR = os.path.join(BASE_DIR, "audio", "letters_wav") # Renamed folder conceptually
REGISTERED_EMBEDDINGS_DIR = os.path.join(BASE_DIR,"audio", "registered_face_embeddings")
REGISTERED_VOICES_DIR = os.path.join(BASE_DIR,"audio", "registered_voices")
TEMP_AUDIO_DIR = os.path.join(BASE_DIR,"audio", "temp_audio")

# --- HARDCODED Bleep File Paths (NOW .wav) ---
# !!! IMPORTANT: REPLACE THESE WITH THE *EXACT* FULL PATHS TO YOUR .wav FILES !!!
HARDCODED_BLEEP_PATH_HIGH = r"C:\Users\HARSH\Desktop\hack36\beep_high.wav"
HARDCODED_BLEEP_PATH_MID  = r"C:\Users\HARSH\Desktop\hack36\beep.wav"
HARDCODED_BLEEP_PATH_LOW  = r"C:\Users\HARSH\Desktop\hack36\beep_low.wav"
# --- End Hardcoded Paths ---

# Ensure base directories exist
os.makedirs(LETTER_AUDIO_DIR, exist_ok=True)
os.makedirs(REGISTERED_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(REGISTERED_VOICES_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Braille Config (Timings remain the same)
BLEEP_DELAY_MS = 350
INTER_LETTER_DELAY_MS = 400
INITIAL_PAUSE_MS = 100
BLEEP_FADE_MS = 5
LEFT_PAN_PD = -0.9
RIGHT_PAN_PD = 0.9

# Voice Auth Config (Remains the same)
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
PREDEFINED_PHRASE = "My voice is my passport, verify me."
VOICE_SIMILARITY_THRESHOLD = 0.60 # Tune this carefully!

# Face Auth Config (Remains the same)
try:
    from deepface import DeepFace
    FACE_MODEL_NAME = "Facenet512"
    FACE_SIMILARITY_THRESHOLD = 0.70 # Tune this carefully!
    FACE_VERIFICATION_DELAY = 2
except ImportError:
    print("WARNING: DeepFace library not found. Face authentication disabled.")
    DeepFace = None

# Braille Patterns (Remain the same)
BRAILLE_PATTERNS = {
    'A': (1, 0, 0, 0, 0, 0), 'B': (1, 1, 0, 0, 0, 0), 'C': (1, 0, 0, 1, 0, 0),
    'D': (1, 0, 0, 1, 1, 0), 'E': (1, 0, 0, 0, 1, 0), 'F': (1, 1, 0, 1, 0, 0),
    'G': (1, 1, 0, 1, 1, 0), 'H': (1, 1, 0, 0, 1, 0), 'I': (0, 1, 0, 1, 0, 0),
    'J': (0, 1, 0, 1, 1, 0), 'K': (1, 0, 1, 0, 0, 0), 'L': (1, 1, 1, 0, 0, 0),
    'M': (1, 0, 1, 1, 0, 0), 'N': (1, 0, 1, 1, 1, 0), 'O': (1, 0, 1, 0, 1, 0),
    'P': (1, 1, 1, 1, 0, 0), 'Q': (1, 1, 1, 1, 1, 0), 'R': (1, 1, 1, 0, 1, 0),
    'S': (0, 1, 1, 1, 0, 0), 'T': (0, 1, 1, 1, 1, 0), 'U': (1, 0, 1, 0, 0, 1),
    'V': (1, 1, 1, 0, 0, 1), 'W': (0, 1, 0, 1, 1, 1), 'X': (1, 0, 1, 1, 0, 1),
    'Y': (1, 0, 1, 1, 1, 1), 'Z': (1, 0, 1, 0, 1, 1),
}
ALPHABET = sorted(BRAILLE_PATTERNS.keys())

# --- TTS Helper Function (Unchanged, still outputs MP3) ---
def speak(text):
    """Generates audio from text and adds player to Streamlit."""
    if not text: return
    print(f"TTS: {text}") # Log what's being spoken
    try:
        text_to_speak = text[:499] # Limit length
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        # Display audio player for TTS (still MP3 format from gTTS)
        st.audio(fp, format='audio/mp3')
    except Exception as e:
        st.error(f"TTS Error ({type(e).__name__}): {e}. Check internet/gTTS.")

# --- Audio Processing Functions (Using WAV) ---

@st.cache_resource # Cache loaded sounds
def load_bleep_sounds_st():
    """Loads the three pitched bleep sounds (.wav) using HARDCODED paths."""
    bleeps = {}
    paths = {
        "high": HARDCODED_BLEEP_PATH_HIGH,
        "mid":  HARDCODED_BLEEP_PATH_MID,
        "low":  HARDCODED_BLEEP_PATH_LOW,
    }
    loaded_correctly = True
    missing_files_msg = []
    status_placeholder = st.empty()
    status_placeholder.info("Loading WAV sound resources...")

    for key, path in paths.items():
        if not os.path.exists(path):
            errmsg = f"Missing bleep WAV file: {os.path.basename(path)} at {path}"
            st.error(errmsg)
            missing_files_msg.append(errmsg)
            loaded_correctly = False
            continue
        try:
            if os.path.getsize(path) == 0:
                 st.error(f"Bleep WAV file '{os.path.basename(path)}' is empty at {path}")
                 loaded_correctly = False
                 continue
            # Load the WAV sound file using from_file (should auto-detect WAV)
            # Or explicitly use from_wav if you are certain
            sound = AudioSegment.from_file(path)
            # sound = AudioSegment.from_wav(path) # Alternative
            bleeps[key] = sound.fade_in(BLEEP_FADE_MS).fade_out(BLEEP_FADE_MS)
            print(f"Successfully loaded {key} bleep from WAV: {path}") # Log success
        except CouldntDecodeError:
             # This might still happen if the WAV file is corrupt or not a standard format
             st.error(f"Error decoding {key} bleep WAV: {os.path.basename(path)}. Is it a valid WAV file?")
             loaded_correctly = False
        except Exception as e:
            st.error(f"Error loading {key} bleep WAV ({os.path.basename(path)}): {type(e).__name__} - {e}")
            loaded_correctly = False

    if loaded_correctly:
        status_placeholder.success("WAV Sound resources loaded successfully.")
        print("All bleep sounds loaded.")
        return bleeps
    else:
        status_placeholder.error("Failed to load one or more bleep WAV sounds. Check paths.")
        print(f"Failed loading bleep WAVs. Errors:\n" + "\n".join(missing_files_msg))
        return None

@st.cache_data
def generate_braille_audio_bytes_st(letter, _bleep_sounds):
    """Generates combined audio using WAV files and exports as WAV bytes."""
    if _bleep_sounds is None:
        st.error("Cannot generate audio: Bleep sounds not loaded.")
        return None

    # --- Load Letter Name (WAV) ---
    letter_file_path = os.path.join(LETTER_AUDIO_DIR, f"{letter}.wav") # Expect .wav now
    if not os.path.exists(letter_file_path):
        print(f"Error: Letter WAV file not found: {letter_file_path}")
        st.error(f"Letter audio file not found: {os.path.basename(letter_file_path)}")
        return None

    try:
        # Use from_file or from_wav
        letter_segment = AudioSegment.from_file(letter_file_path)
        # letter_segment = AudioSegment.from_wav(letter_file_path) # Alternative
    except Exception as e:
        print(f"Error loading letter WAV segment {letter}: {e}")
        st.error(f"Error loading letter audio: {e}")
        return None

    # --- Combine Audio (Logic is the same) ---
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
                 print(f"Warning: Could not find loaded bleep sound for key '{bleep_key}' for dot {dot_number}")
        bleep_sequence += processed_bleep
        padding_needed = max(0, BLEEP_DELAY_MS - len(processed_bleep))
        bleep_sequence += AudioSegment.silent(duration=padding_needed)

    final_audio += bleep_sequence

    # --- Export to buffer as WAV ---
    try:
        buffer = io.BytesIO()
        final_audio.export(buffer, format="wav") # Export as WAV
        print(f"Successfully exported combined WAV for letter {letter}")
        return buffer.getvalue()
    except Exception as e:
        print(f"Error exporting combined WAV audio to memory for letter {letter}: {e}")
        st.error(f"Error exporting audio: {e}")
        return None


# --- Other functions (record_audio_st, extract_mel_spectrogram_st, auth functions) remain the same ---
# They already work with WAV for temporary recordings or don't depend on the bleep/letter format.
def record_audio_st(filename, duration=DURATION, sr=SAMPLE_RATE):
    # (Code remains the same, uses updated speak function)
    speak(f"Please get ready to speak the phrase: '{PREDEFINED_PHRASE}'")
    time.sleep(1.5)
    speak("Recording starts in 3...")
    time.sleep(1)
    speak("2...")
    time.sleep(1)
    speak("1...")
    time.sleep(1)
    speak("Recording now!")
    try:
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        sf.write(filename, recording, sr)
        speak("Recording finished.")
        return filename
    except Exception as e:
        st.error(f"Error during recording: {e}")
        speak("Sorry, there was an error during recording.")
        return None

def extract_mel_spectrogram_st(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # (Code remains the same)
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        if sr_loaded != sr:
            st.warning(f"Audio loaded with sample rate {sr_loaded}, resampling to {sr}.")
        y, index = librosa.effects.trim(y, top_db=25)
        if len(y) == 0: return None
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        target_len = int(np.ceil(DURATION * sr / hop_length)) + 1
        if log_mel_spec.shape[1] < target_len:
            pad_width = target_len - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='minimum')
        else:
            log_mel_spec = log_mel_spec[:, :target_len]
        return log_mel_spec
    except Exception as e:
        st.error(f"Error processing audio {audio_path}: {e}")
        return None

@st.cache_resource
def load_registered_voices_st():
     # (Code remains the same)
    registered_data = {}
    loaded_count = 0
    for filename in os.listdir(REGISTERED_VOICES_DIR):
        if filename.endswith(".npy"):
            user_id = os.path.splitext(filename)[0]
            filepath = os.path.join(REGISTERED_VOICES_DIR, filename)
            try:
                registered_data[user_id] = np.load(filepath)
                loaded_count += 1
            except Exception as e:
                st.warning(f"Error loading voice for {user_id}: {e}")
    return registered_data

@st.cache_resource
def load_registered_embeddings_st():
    # (Code remains the same)
    known_face_embeddings = []
    known_face_names = []
    if DeepFace is None: return known_face_embeddings, known_face_names
    loaded_count = 0
    for filename in os.listdir(REGISTERED_EMBEDDINGS_DIR):
        if filename.endswith(".npy"):
            filepath = os.path.join(REGISTERED_EMBEDDINGS_DIR, filename)
            user_id = os.path.splitext(filename)[0]
            try:
                embedding = np.load(filepath).flatten()
                known_face_embeddings.append(embedding)
                known_face_names.append(user_id)
                loaded_count+=1
            except Exception as e:
                st.warning(f"Error loading embedding for {user_id}: {e}")
    return known_face_embeddings, known_face_names

def run_face_verification_window(known_embeddings, known_names):
    # (Code remains the same)
    if DeepFace is None:
        st.error("DeepFace library not available.")
        return False, None
    if not known_embeddings:
        st.warning("No registered faces found for verification.")
        return False, None
    st.info("Face verification window will open. Please look at the camera.")
    speak("Face verification window opening. Please look at the camera until access is granted or denied, or press Q in the window to quit.")
    time.sleep(2)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        st.error("Could not open webcam.")
        speak("Sorry, could not open the webcam.")
        return False, None
    last_match_time = 0
    verified_user = None
    access_granted_tentative = False
    verification_success = False
    while True:
        ret, frame = video_capture.read()
        if not ret: break
        display_frame = frame.copy()
        live_embedding, facial_area = None, None
        try:
            embedding_objs = DeepFace.represent(img_path=frame, model_name=FACE_MODEL_NAME, enforce_detection=True, detector_backend='opencv')
            if embedding_objs:
                live_embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
                facial_area = embedding_objs[0]["facial_area"]
        except ValueError: pass
        except Exception as e: print(f"Error during DeepFace represent: {e}")
        best_match_name = "Unknown"
        best_similarity = -1
        if live_embedding is not None and facial_area is not None:
            live_embedding_reshaped = live_embedding.reshape(1, -1)
            similarities = cosine_similarity(live_embedding_reshaped, np.array(known_embeddings))
            best_match_index = np.argmax(similarities[0])
            best_similarity = similarities[0][best_match_index]
            if best_similarity >= FACE_SIMILARITY_THRESHOLD:
                best_match_name = known_names[best_match_index]
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            box_color = (0, 0, 255) # Red default
            label = f"Unknown ({best_similarity:.2f})"
            if best_match_name != "Unknown":
                label = f"Verifying: {best_match_name} ({best_similarity:.2f})"
                box_color = (255, 255, 0) # Cyan
                if not access_granted_tentative:
                    access_granted_tentative = True
                    last_match_time = time.time()
                    verified_user = best_match_name
                elif verified_user == best_match_name:
                    if time.time() - last_match_time > FACE_VERIFICATION_DELAY:
                        box_color = (0, 255, 0) # Green
                        label = f"Access Granted: {best_match_name}"
                        verification_success = True
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
                        cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        cv2.imshow('Face Verification - Press Q to Close', display_frame)
                        cv2.waitKey(1500)
                        break
                else: # Different user
                    last_match_time = time.time()
                    verified_user = best_match_name
                    access_granted_tentative = True
            else: # Unknown
                 label = f"Unknown (Sim: {best_similarity:.2f})"
                 access_granted_tentative = False
                 verified_user = None
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        else: # No face
            cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            access_granted_tentative = False
            verified_user = None
        cv2.imshow('Face Verification - Press Q to Close', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    video_capture.release()
    cv2.destroyAllWindows()
    if verification_success:
        speak(f"Face verification successful. Welcome {verified_user}.")
        return True, verified_user
    else:
        speak("Face verification failed or was cancelled.")
        return False, None

# --- Streamlit App UI ---

# Initialize session state
# --- Streamlit App ---

# Initialize session state (Ensure ALL keys exist)
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state: # Check specifically for user_id
    st.session_state.user_id = None
if 'current_letter_index' not in st.session_state:
    st.session_state.current_letter_index = 0
if 'bleep_sounds' not in st.session_state:
    st.session_state.bleep_sounds = None
if 'key_processed' not in st.session_state:
    st.session_state.key_processed = True # Start as True

# --- Load Bleep Sounds ONCE ---
# This uses the function with hardcoded paths now
# The result is cached using @st.cache_resource
if st.session_state.bleep_sounds is None:
     st.session_state.bleep_sounds = load_bleep_sounds_st()

# --- Authentication Gate ---
# ... (rest of the script) ...# --- Authentication Gate ---
if not st.session_state.authenticated:
    st.title("Welcome to the Accessible Braille Learner")
    speak("Welcome to the Accessible Braille Learner. Please authenticate to continue.")

    auth_method = st.radio("Choose Authentication Method:", ("Face Recognition", "Voice Recognition"), horizontal=True, label_visibility="visible")
    st.markdown(f"Selected: **{auth_method}**")

    if auth_method == "Face Recognition":
        st.header("Face Authentication")
        speak("Press the button below to start face verification.")
        if st.button("Start Face Verification"):
            if DeepFace is None:
                 st.error("Face Auth unavailable.")
                 speak("Sorry, face authentication is not available.")
            else:
                 known_face_embeddings, known_face_names = load_registered_embeddings_st()
                 success, user_id = run_face_verification_window(known_face_embeddings, known_face_names)
                 if success:
                     st.session_state.authenticated = True
                     st.session_state.user_id = user_id
                     st.success(f"Authentication successful! Welcome {user_id}")
                     st.rerun()
                 else:
                     st.error("Face Authentication Failed.")

    elif auth_method == "Voice Recognition":
        st.header("Voice Authentication")
        speak(f"Please be ready to say the phrase: {PREDEFINED_PHRASE}. Press the button below to start.")
        if st.button("Start Voice Verification"):
            registered_voices = load_registered_voices_st()
            if not registered_voices:
                 st.warning("No registered voices found.")
                 speak("There are no registered voices to compare against.")
            else:
                 verify_audio_path = os.path.join(TEMP_AUDIO_DIR, f"verify_{st.session_state.user_id or 'unknown'}_{int(time.time())}.wav")
                 if record_audio_st(verify_audio_path):
                     verify_spec = extract_mel_spectrogram_st(verify_audio_path)
                     if verify_spec is not None:
                         best_match_user = None
                         highest_similarity = -1
                         verify_spec_flat = verify_spec.flatten().reshape(1, -1)
                         with st.spinner("Comparing voice..."):
                             for user_id, reg_spec in registered_voices.items():
                                 if reg_spec.shape == verify_spec.shape:
                                     reg_spec_flat = reg_spec.flatten().reshape(1, -1)
                                     similarity = cosine_similarity(verify_spec_flat, reg_spec_flat)[0][0]
                                     print(f"Voice sim with {user_id}: {similarity:.4f}")
                                     if similarity > highest_similarity:
                                         highest_similarity = similarity
                                         best_match_user = user_id
                         if highest_similarity >= VOICE_SIMILARITY_THRESHOLD:
                             st.success(f"Voice Verification Successful! Welcome {best_match_user} (Similarity: {highest_similarity:.4f})")
                             speak(f"Voice verification successful. Welcome {best_match_user}.")
                             st.session_state.authenticated = True
                             st.session_state.user_id = best_match_user
                             st.rerun()
                         else:
                             st.error(f"Voice Verification Failed. Highest similarity ({highest_similarity:.4f}) is below threshold.")
                             speak("Voice verification failed. Access denied.")
                     else:
                         st.error("Could not process recorded voice.")
                         speak("Sorry, could not process the recorded voice.")
                     if os.path.exists(verify_audio_path): os.remove(verify_audio_path)

    st.divider()
    st.subheader("Need to Register?")
    st.info("Registration must be done using the original scripts from the command line.")
    speak("If you need to register your face or voice, please use the original scripts provided.")

# --- Main Application (If Authenticated) ---
else:
    st.sidebar.success(f"Authenticated as: {st.session_state.user_id}")
    if st.sidebar.button("Logout"):
        speak("Logging out.")
        for key in list(st.session_state.keys()):
             if key not in ['bleep_sounds']:
                 del st.session_state[key]
        st.session_state.authenticated = False
        st.rerun()

    st.title("Braille Audio Learner")
    speak("Braille Audio Learner. Use the buttons to navigate letters.")

    if st.session_state.bleep_sounds is None:
        st.error("Bleep sounds could not be loaded. Braille learning disabled.")
        st.warning("Please ensure beep.wav, beep_high.wav, and beep_low.wav exist at the specified hardcoded paths.")
        speak("Error: Bleep sounds are missing. Braille learning is disabled.")
    else:
        current_index = st.session_state.current_letter_index
        current_letter = ALPHABET[current_index]
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⬅️ Previous Letter", key="prev_letter"):
                st.session_state.current_letter_index = (current_index - 1) % len(ALPHABET)
                st.rerun()

        with col2:
             st.markdown(f"<h1 style='text-align: center; font-size: 5em;'>{current_letter}</h1>", unsafe_allow_html=True)
             speak(f"Current letter: {current_letter}")

        with col3:
            if st.button("Next Letter ➡️", key="next_letter"):
                st.session_state.current_letter_index = (current_index + 1) % len(ALPHABET)
                st.rerun()

        st.divider()
        st.subheader("Braille Sound")

        # Generate audio bytes for the current letter (using WAV now)
        audio_bytes = generate_braille_audio_bytes_st(current_letter, st.session_state.bleep_sounds)

        if audio_bytes:
            try:
                # Display the audio player with WAV format
                st.audio(audio_bytes, format='audio/wav')
            except Exception as e:
                st.error(f"Error displaying WAV audio player: {e}")
                speak("Sorry, there was an error displaying the audio player.")
        else:
            st.error(f"Could not generate audio for letter '{current_letter}'.")
            st.warning("Check console logs for errors regarding letter or bleep WAV files.")
            speak(f"Sorry, could not generate the audio for letter {current_letter}.")

st.markdown("---")
st.info("Developed as an accessible learning tool.")