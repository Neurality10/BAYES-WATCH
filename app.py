import streamlit as st
# --- >>> set_page_config() MUST be the first Streamlit command <<< ---
st.set_page_config(
    page_title="Accessible Learning Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- End Placement ---

import os
import time
import sys
import logging
from pathlib import Path
import tempfile
import json
import requests # Needed for Text-to-Braille

# --- Project Helper Module Imports ---
try:
    import dependencies # Import dependency checker first
except ImportError as e:
    st.error(f"Fatal Error: Failed to import dependency checker: {e}. Application cannot start.")
    st.stop()

# === Run Dependency Check Early ===
# Add ALL required packages for ALL features here
ALL_REQUIRED_PACKAGES = [
    # Auth/Braille
    'sounddevice', 'soundfile', 'librosa', 'opencv-python', 'scikit-learn', 'pydub', 'pygame',
    # Doc Processor V2
    'requests', 'pillow', 'docling', 'python-dateutil', 'python-dotenv', 'gTTS', 'pandas', 'openpyxl',
    # Voice Exam Taker
    'SpeechRecognition', 'streamlit-mic-recorder',
    # General
    'numpy' # Already a dependency of many others, but good to list
]
# Optional (like DeepFace) are checked within their helpers or here if critical
if not dependencies.install_and_check(ALL_REQUIRED_PACKAGES):
     st.error("❌ Critical dependencies failed to install or verify. Application cannot continue.")
     st.stop()
else:
     st.sidebar.success("✅ Core dependencies verified.")
# === End Dependency Check ===


try:
    import auth_braille_helpers as auth_helpers
    log_auth = logging.getLogger(auth_helpers.__name__) # Get logger from helper
except ImportError as e:
    st.error(f"Fatal Error: Failed to import authentication/braille helpers: {e}.")
    st.stop()

try:
    import doc_processor_v2_helpers as doc_helpers
    from doc_processor_v2_helpers import DOC_V2_MODULES_LOADED
    log_doc = logging.getLogger(doc_helpers.__name__) # Get logger from helper
except ImportError as e:
    st.error(f"Fatal Error: Failed to import document processing helpers: {e}.")
    st.stop()

# --- NEW: Import Voice Exam Helpers ---
try:
    import voice_exam_helpers as ve_helpers
    from voice_exam_helpers import SPEECH_RECOGNITION_AVAILABLE # Import check flag
    log_ve = logging.getLogger(ve_helpers.__name__)
except ImportError as e:
    st.error(f"Fatal Error: Failed to import voice exam helpers: {e}.")
    # Decide if the app should stop if voice exam fails to load
    # For now, let it continue but the feature will be disabled/error out
    ve_helpers = None # Set to None so checks fail later
    SPEECH_RECOGNITION_AVAILABLE = False
    log_ve = logging.getLogger("voice_exam_dummy") # Dummy logger
    log_ve.error(f"Failed to import voice_exam_helpers: {e}")


# --- Logging Setup (Main App) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
log = logging.getLogger(__name__)
log.info("Main application starting...")

# --- Configuration (App Specific & Aggregated) ---
BRAILLE_API_URL = "https://brailletranslate.vercel.app/api/translate"
CWD = Path.cwd() # Current working directory

# --- Aggregate Configuration ---
APP_CONFIG = {
    # --- Auth/Braille Settings ---
    "APP1_BASE_DIR": str(auth_helpers.APP1_BASE_DIR),
    "LETTER_AUDIO_DIR": str(auth_helpers.LETTER_AUDIO_DIR),
    "REGISTERED_EMBEDDINGS_DIR": str(auth_helpers.REGISTERED_EMBEDDINGS_DIR),
    "REGISTERED_VOICES_DIR": str(auth_helpers.REGISTERED_VOICES_DIR),
    "TEMP_AUDIO_DIR": str(auth_helpers.TEMP_AUDIO_DIR),
    "HARDCODED_BLEEP_PATH_HIGH": str(auth_helpers.APP1_BASE_DIR / auth_helpers.BEEP_HIGH_FILENAME),
    "HARDCODED_BLEEP_PATH_MID": str(auth_helpers.APP1_BASE_DIR / auth_helpers.BEEP_MID_FILENAME),
    "HARDCODED_BLEEP_PATH_LOW": str(auth_helpers.APP1_BASE_DIR / auth_helpers.BEEP_LOW_FILENAME),
    "BLEEP_DELAY_MS": auth_helpers.BLEEP_DELAY_MS,
    "INTER_LETTER_DELAY_MS": auth_helpers.INTER_LETTER_DELAY_MS,
    "INITIAL_PAUSE_MS": auth_helpers.INITIAL_PAUSE_MS,
    "BLEEP_FADE_MS": auth_helpers.BLEEP_FADE_MS,
    "LEFT_PAN_PD": auth_helpers.LEFT_PAN_PD,
    "RIGHT_PAN_PD": auth_helpers.RIGHT_PAN_PD,
    "SAMPLE_RATE": auth_helpers.SAMPLE_RATE,
    "VOICE_REC_DURATION": auth_helpers.VOICE_REC_DURATION,
    "N_MELS": auth_helpers.N_MELS,
    "N_FFT": auth_helpers.N_FFT,
    "HOP_LENGTH": auth_helpers.HOP_LENGTH,
    "PREDEFINED_PHRASE": auth_helpers.PREDEFINED_PHRASE,
    "VOICE_SIMILARITY_THRESHOLD": auth_helpers.VOICE_SIMILARITY_THRESHOLD,
    "FACE_MODEL_NAME": auth_helpers.FACE_MODEL_NAME,
    "FACE_SIMILARITY_THRESHOLD": auth_helpers.FACE_SIMILARITY_THRESHOLD,
    "FACE_VERIFICATION_DELAY": auth_helpers.FACE_VERIFICATION_DELAY,

    # --- Doc Processor V2 Settings ---
    "APP2_INPUT_DIR": str(CWD / doc_helpers.APP2_INPUT_DIR_NAME),
    "APP2_OUTPUT_DIR": str(CWD / doc_helpers.APP2_OUTPUT_DIR_NAME),
    "APP2_TTS_OUTPUT_DIR": str(CWD / doc_helpers.APP2_TTS_OUTPUT_DIR_NAME),
    "APP2_DOCLING_DEVICE": doc_helpers.APP2_DOCLING_DEVICE,
    "APP2_IMAGE_SCALE": doc_helpers.APP2_IMAGE_SCALE,
    "APP2_TTS_ENGINE": doc_helpers.APP2_TTS_ENGINE,
    "APP2_TTS_LANG": doc_helpers.APP2_TTS_LANG,
    "APP2_TTS_CHUNK_SIZE": doc_helpers.APP2_TTS_CHUNK_SIZE,
    "APP2_SUPPORTED_EXTENSIONS": list(doc_helpers.APP2_SUPPORTED_EXTENSIONS),
    "CLOUDINARY_CLOUD_NAME": doc_helpers.CLOUDINARY_CLOUD_NAME,
    "CLOUDINARY_API_KEY": doc_helpers.CLOUDINARY_API_KEY[:4] + "***" if doc_helpers.CLOUDINARY_API_KEY else None,
    "CLOUDINARY_API_SECRET": doc_helpers.CLOUDINARY_API_SECRET[:4] + "***" if doc_helpers.CLOUDINARY_API_SECRET else None,
    "OPENROUTER_API_KEY": doc_helpers.OPENROUTER_API_KEY[:7] + "***" if doc_helpers.OPENROUTER_API_KEY else None,
    "OPENROUTER_API_BASE": doc_helpers.OPENROUTER_API_BASE,
    "QWEN_MODEL": doc_helpers.QWEN_MODEL,

    # --- Voice Exam Taker Settings (Merged from its config.py) ---
    # !!! IMPORTANT: User must set this path correctly !!!
    "VE_PIPELINE_OUTPUT_BASE_DIR": r"C:\Darsh\Hack\pipeline_output", # <<< MODIFY THIS PATH !!!
    # Define other VE paths relative to CWD (Current Working Directory)
    "VE_SUBMISSIONS_DIR": str(CWD / "submissions_voice_exam"),
    "VE_TTS_OUTPUT_DIR": str(CWD / "tts_output_voice_exam"),
    "VE_EXAM_DURATION_MINUTES": 60,
    "VE_TTS_LANGUAGE": "en-US", # Separate from Doc Proc TTS lang if needed
    "VE_SHOW_DEBUG_INFO": True, # Or get from env var
    "VE_VOICE_ENGINE_LANGUAGE": "en-US",
    "VE_VOICE_ENGINE_PAUSE_THRESHOLD": 0.8,
    "VE_VOICE_ENGINE_DYNAMIC_ENERGY": True,
    "VE_VOICE_ENGINE_ENERGY_THRESHOLD": 400,
    "VE_EXAM_JSON_GLOB_PATTERN": os.path.join("*", "*_transcripts.json"), # Relative to run folder
    "VE_EXAM_AUDIO_DIR_GLOB_PATTERN": os.path.join("*", "audio_gtts"), # Relative to run folder
    "VE_EXAM_RUN_DIR_PREFIX": "streamlit_run_", # Prefix for pipeline output folders

    # --- Text to Braille ---
    "BRAILLE_API_URL": BRAILLE_API_URL,
}

# --- Initialize Session State ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'user_id' not in st.session_state: st.session_state.user_id = None
if 'bleep_sounds' not in st.session_state:
    st.session_state.bleep_sounds = auth_helpers.load_bleep_sounds_st()
    if st.session_state.bleep_sounds is None:
        log.error("Bleep sounds failed to load during initialization.")
        st.error("Critical audio resources failed to load. Braille Learner disabled.")
if 'current_page' not in st.session_state: st.session_state.current_page = "Braille Learner"
if 'braille_letter_index' not in st.session_state: st.session_state.braille_letter_index = 0
# V2 Doc State
if 'doc_v2_status' not in st.session_state: st.session_state.doc_v2_status = "Idle"
if 'doc_v2_error' not in st.session_state: st.session_state.doc_v2_error = ""
if 'doc_v2_success' not in st.session_state: st.session_state.doc_v2_success = None
if 'doc_v2_audio' not in st.session_state: st.session_state.doc_v2_audio = []
if 'doc_v2_md_file' not in st.session_state: st.session_state.doc_v2_md_file = None
if 'doc_v2_file_name' not in st.session_state: st.session_state.doc_v2_file_name = None
# Text to Braille State
if 'text2braille_input' not in st.session_state: st.session_state.text2braille_input = ""
if 'text2braille_output' not in st.session_state: st.session_state.text2braille_output = ""
# Voice Exam State (Initialized within helper on page load)
# Config
if 'app_config' not in st.session_state: st.session_state.app_config = APP_CONFIG


# --- === Authentication Gate === ---
if not st.session_state.get("authenticated", False):
    st.title("Welcome - Accessible Learning Suite")
    st.markdown("Please authenticate using one of the methods below.")

    # Determine available auth methods
    auth_options = ["Voice Recognition"] # Voice is always available
    if auth_helpers.DeepFace is not None:
        auth_options.insert(0, "Face Recognition") # Add face if available

    auth_method = st.radio("Authentication Method:", auth_options, horizontal=True, key="auth_radio")

    # --- Face Recognition ---
    if auth_method == "Face Recognition" and auth_helpers.DeepFace is not None:
        if st.button("Start Face Verification", key="face_btn"):
            known_embeddings, known_names = auth_helpers.load_registered_embeddings_st()
            if not known_embeddings:
                st.warning("No faces registered.")
            else:
                success, user_id = auth_helpers.run_face_verification_window(known_embeddings, known_names)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id
                    st.success(f"Authenticated as {user_id}!")
                    log.info(f"User {user_id} authenticated via Face Recognition.")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error("Face Authentication Failed.")
                    log.warning("Face authentication failed.")

    # --- Voice Recognition ---
    elif auth_method == "Voice Recognition":
        if st.button("Start Voice Verification", key="voice_btn"):
            registered_voices = auth_helpers.load_registered_voices_st()
            if not registered_voices:
                st.warning("No voices registered.")
            else:
                temp_verify_path = auth_helpers.TEMP_AUDIO_DIR / f"verify_{int(time.time())}.wav"
                recorded_file = auth_helpers.record_audio_st(str(temp_verify_path))
                if recorded_file:
                    current_spectrogram = auth_helpers.extract_mel_spectrogram_st(recorded_file)
                    if current_spectrogram is not None:
                        best_match_user, highest_similarity = auth_helpers.verify_voice_st(
                            current_spectrogram, registered_voices
                        ) # Use helper
                        if best_match_user:
                            st.session_state.authenticated = True
                            st.session_state.user_id = best_match_user
                            st.success(f"Authenticated as {best_match_user} (Similarity: {highest_similarity:.2f})")
                            log.info(f"User {best_match_user} authenticated via Voice.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error(f"Voice Authentication Failed (Similarity: {highest_similarity:.2f})")
                            log.warning(f"Voice authentication failed.")
                    else: st.error("Could not process recorded audio.")
                    # Cleanup handled in verify_voice_st or here if needed
                    if Path(recorded_file).exists(): os.remove(recorded_file)
                else: st.error("Audio recording failed.")

    st.info("Note: User registration (face/voice) needs to be done separately.", icon="ℹ️")

# --- === Main Application (Authenticated) === ---
else:
    # --- Sidebar ---
    st.sidebar.success(f"Authenticated as: {st.session_state.get('user_id', 'Unknown')}")
    if st.sidebar.button("Logout", key="logout_btn"):
        log.info(f"User {st.session_state.get('user_id', 'Unknown')} logging out.")
        keys_to_clear = list(st.session_state.keys())
        persistent_keys = ['app_config', 'bleep_sounds']
        for key in keys_to_clear:
            if key not in persistent_keys:
                try: del st.session_state[key]
                except KeyError: pass
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.success("Logged out.")
        time.sleep(1)
        st.rerun()

    # --- Navigation ---
    app_config = st.session_state.app_config
    bleep_sounds = st.session_state.bleep_sounds

    # Define available pages
    page_options = ["Braille Learner", "Text to Braille"] # Start with universally available pages

    # Conditionally add Doc Processor V2
    if DOC_V2_MODULES_LOADED:
        page_options.append("Document Processor V2")
    else:
         st.sidebar.warning("Document Processor V2 disabled (modules failed).")

    # --- NEW: Conditionally add Voice Exam Taker ---
    if ve_helpers is not None and SPEECH_RECOGNITION_AVAILABLE:
         page_options.append("Voice Exam Taker")
    elif ve_helpers is None:
         st.sidebar.error("Voice Exam Taker disabled (module load failed).")
    elif not SPEECH_RECOGNITION_AVAILABLE:
         st.sidebar.warning("Voice Exam Taker disabled (SpeechRecognition unavailable).")


    # Ensure default page exists in available options
    default_page = st.session_state.get("current_page", page_options[0])
    if default_page not in page_options:
        default_page = page_options[0]

    current_selection = st.sidebar.radio(
        "Select Feature",
        page_options,
        key="page_nav",
        index=page_options.index(default_page)
    )
    st.session_state.current_page = current_selection

    # --- === Page Rendering === ---

    # --- Braille Learner Page ---
    if st.session_state.current_page == "Braille Learner":
        st.title("🔡 Braille Audio Learner")
        st.markdown("Learn Braille patterns via audio. Use headphones.")

        if bleep_sounds is None:
             st.error("Audio resources missing. Cannot generate Braille audio.")
             st.stop()

        idx = st.session_state.braille_letter_index
        current_letter = auth_helpers.ALPHABET[idx]

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", key="b_prev", use_container_width=True):
                st.session_state.braille_letter_index = (idx - 1) % len(auth_helpers.ALPHABET)
                st.rerun()
        with col2:
            st.markdown(f"<h1 style='text-align: center; font-size: 6em; margin-bottom: 0;'>{current_letter}</h1>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", key="b_next", use_container_width=True):
                st.session_state.braille_letter_index = (idx + 1) % len(auth_helpers.ALPHABET)
                st.rerun()
        st.divider()
        st.subheader(f"Audio Representation for '{current_letter}'")
        audio_bytes = auth_helpers.generate_braille_audio_bytes_st(current_letter, bleep_sounds)
        if audio_bytes: st.audio(audio_bytes, format='audio/wav')
        else: st.error("Failed to generate audio.")

    # --- Document Processor V2 Page ---
    elif st.session_state.current_page == "Document Processor V2" and DOC_V2_MODULES_LOADED:
        st.title("📄➡️🔊 Document Processor V2")
        st.markdown("Upload a document (.pdf, .docx) for content extraction and audio narration.")

        try:
             docling_pipe_v2, tts_gen_v2 = doc_helpers.initialize_doc_processor_v2_pipelines(app_config)
        except Exception as init_err:
             st.error(f"V2 pipeline initialization error: {init_err}")
             log.error("V2 Pipeline init failed.", exc_info=True)
             st.stop()

        if docling_pipe_v2 is None: st.error("Doc Processing component failed initialization.")
        if app_config.get("APP2_TTS_ENGINE") and tts_gen_v2 is None: st.warning("TTS component failed initialization.")

        st.divider()
        st.subheader("1. Upload Document")
        supported_ext_list = app_config.get("APP2_SUPPORTED_EXTENSIONS", [])
        supported_types = [ext.lstrip('.') for ext in supported_ext_list]
        uploaded_file = st.file_uploader(
            f"Choose document ({', '.join(supported_ext_list)})",
            type=supported_types, key="uploader_v2"
        )

        if uploaded_file is not None and docling_pipe_v2 is not None:
            if st.button(f"Process '{uploaded_file.name}'", key="process_doc_v2_btn"):
                st.session_state.doc_v2_status = "Running"; st.session_state.doc_v2_error = ""
                st.session_state.doc_v2_success = None; st.session_state.doc_v2_audio = []
                st.session_state.doc_v2_md_file = None; st.session_state.doc_v2_file_name = uploaded_file.name

                input_dir_path = Path(app_config["APP2_INPUT_DIR"])
                tmp_path = input_dir_path / uploaded_file.name
                try:
                    input_dir_path.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(uploaded_file.getvalue())
                    log.info(f"Saved uploaded file to: {tmp_path}")
                except Exception as save_err:
                    st.error(f"Error saving uploaded file: {save_err}")
                    st.session_state.doc_v2_status = "Error"; st.stop()

                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    start_time = time.time()
                    doc_processing_ok = False; tts_processing_ok = False

                    try: # Calculate Paths
                        output_base = Path(app_config["APP2_OUTPUT_DIR"])
                        tts_output_base = Path(app_config["APP2_TTS_OUTPUT_DIR"])
                        paths = doc_helpers.get_doc_v2_paths(output_base, tts_output_base, tmp_path.stem)
                        paths['docling_output'].mkdir(parents=True, exist_ok=True)
                    except Exception as path_err:
                        st.error(f"Internal Error calculating paths: {path_err}")
                        st.session_state.doc_v2_status = "Error"; st.stop()

                    try: # Document Parsing
                        log.info(f"Calling V2 process_document: Input='{tmp_path}', OutputDir='{paths['docling_output']}'")
                        processing_result = docling_pipe_v2.process_document(
                            doc_path=tmp_path, doc_output_dir=paths['docling_output']
                        )
                        if isinstance(processing_result, dict):
                            doc_processing_ok = processing_result.get('success', False)
                            if not doc_processing_ok:
                                 err_msg = processing_result.get('status', 'Unknown Error')
                                 st.session_state.doc_v2_error += f"\nDocling Error: {err_msg}"
                        else: doc_processing_ok = False
                        if doc_processing_ok and paths["book_like_md"].exists():
                            st.session_state.doc_v2_md_file = paths["book_like_md"]
                        elif doc_processing_ok: st.session_state.doc_v2_error += "\nWarning: Main Markdown file not found."
                    except Exception as doc_err:
                        log.exception("Error during Document Parsing step.")
                        st.session_state.doc_v2_error += f"\nCritical Error during Parsing: {doc_err}"
                        doc_processing_ok = False

                    tts_enabled = app_config.get("APP2_TTS_ENGINE") and tts_gen_v2
                    if doc_processing_ok and tts_enabled: # TTS Generation
                        md_file_path = st.session_state.doc_v2_md_file
                        if md_file_path and md_file_path.exists():
                            try:
                                paths["tts_audio_dir"].mkdir(parents=True, exist_ok=True)
                                audio_file_paths = tts_gen_v2.process_markdown_to_speech(
                                    str(md_file_path), str(paths["tts_audio_dir"]),
                                    app_config.get("APP2_TTS_CHUNK_SIZE")
                                )
                                st.session_state.doc_v2_audio = [Path(p) for p in audio_file_paths if p]
                                tts_processing_ok = True
                            except Exception as tts_err:
                                log.exception("Error during V2 TTS step.")
                                st.session_state.doc_v2_error += f"\nError during TTS: {tts_err}"
                        else: tts_processing_ok = False # Mark TTS as failed if MD missing
                    elif doc_processing_ok: tts_processing_ok = True # OK if TTS not enabled
                    else: tts_processing_ok = False # Failed if doc parsing failed

                    st.session_state.doc_v2_success = doc_processing_ok and tts_processing_ok
                    st.session_state.doc_v2_status = "Finished"
                    log.info(f"V2 Processing complete in {time.time() - start_time:.2f}s. Success: {st.session_state.doc_v2_success}")

                if tmp_path.exists(): os.remove(tmp_path) # Cleanup temp input
                st.rerun()

        # Display V2 Results
        if st.session_state.doc_v2_status == "Finished":
            st.divider()
            st.header(f"Results for: {st.session_state.doc_v2_file_name}")
            if st.session_state.doc_v2_success: st.success("✅ Document processed successfully!")
            else: st.error("❌ Processing encountered errors.")
            if st.session_state.doc_v2_error:
                with st.expander("Show Errors/Warnings", expanded=not st.session_state.doc_v2_success):
                    st.warning(st.session_state.doc_v2_error.strip())

            try: # Get paths again for display
                file_name = st.session_state.doc_v2_file_name
                if not file_name: raise ValueError("Filename missing.")
                paths = doc_helpers.get_doc_v2_paths(
                    Path(app_config["APP2_OUTPUT_DIR"]), Path(app_config["APP2_TTS_OUTPUT_DIR"]),
                    Path(file_name).stem
                )
            except Exception as path_err: st.error(f"Error retrieving result paths: {path_err}"); paths = {}

            # Display Markdown
            st.subheader("📖 Extracted Content (Markdown)")
            md_file = st.session_state.get("doc_v2_md_file")
            if md_file and isinstance(md_file, Path) and md_file.exists():
                st.markdown(f"Content saved to: `{md_file}`")
                try:
                    with st.expander("Preview Content"):
                        content = md_file.read_text(encoding='utf-8')
                        st.markdown(content[:10000] + ("..." if len(content) > 10000 else ""))
                except Exception as read_err: st.error(f"Error reading preview: {read_err}")
            elif st.session_state.doc_v2_success: st.warning("Markdown file missing.")
            else: st.info("Markdown not generated due to errors.")

            # Display Audio
            st.subheader("🔊 Generated Audio")
            audio_files = st.session_state.get('doc_v2_audio', [])
            if audio_files and paths.get('tts_audio_dir'):
                st.write(f"{len(audio_files)} audio file(s) generated in `{paths['tts_audio_dir']}`:")
                audio_files.sort(key=lambda x: x.name)
                for af_path in audio_files:
                    if isinstance(af_path, Path) and af_path.exists():
                        try: st.write(f"**{af_path.name}**"); st.audio(str(af_path))
                        except Exception as audio_err: st.error(f"Error displaying {af_path.name}: {audio_err}")
                    elif isinstance(af_path, Path): st.warning(f"Audio file listed but not found: {af_path.name}")
            else: # No audio files
                if app_config.get("APP2_TTS_ENGINE") and st.session_state.doc_v2_success: st.warning("TTS enabled, but no audio files found.")
                elif app_config.get("APP2_TTS_ENGINE"): st.info("Audio generation skipped due to errors.")
                else: st.info("Audio generation disabled.")

            # Display Other Outputs Location
            st.subheader("🗂️ Other Outputs")
            if paths.get('root'): st.markdown(f"Detailed outputs saved in: `{paths['root']}`")
            else: st.warning("Could not determine detailed output directory.")

        elif uploaded_file is None and st.session_state.get("doc_v2_status", "Idle") == "Idle":
            st.info("Upload a document to begin processing.")

    # --- Text to Braille Page ---
    elif st.session_state.current_page == "Text to Braille":
        st.title("🅰️➡️⠿ Text to Braille Translator")
        st.markdown("Enter text to translate into Braille Unicode characters.")
        input_text = st.text_area("Enter text:", height=150, key="t2b_input", value=st.session_state.text2braille_input)
        st.session_state.text2braille_input = input_text

        if st.button("Translate to Braille", key="t2b_btn"):
            if not input_text.strip(): st.warning("Please enter text.")
            else:
                with st.spinner("Translating..."):
                    try:
                        response = requests.get(BRAILLE_API_URL, params={"text": input_text}, timeout=20)
                        response.raise_for_status()
                        st.session_state.text2braille_output = response.text
                    except requests.exceptions.Timeout: st.error("API timed out.")
                    except requests.exceptions.RequestException as e: st.error(f"API connection error: {e}")
                    except Exception as e: st.error(f"Unexpected error: {e}")

        if st.session_state.text2braille_output:
            st.subheader("Braille Output:")
            # Braille font styling
            st.markdown(
                f"""<div style="font-size: 1.8em; font-family: 'Segoe UI Symbol', sans-serif; border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9; line-height: 1.6;">{st.session_state.text2braille_output}</div>""",
                unsafe_allow_html=True
            )
            if st.button("Clear Output", key="clear_braille_btn"):
                st.session_state.text2braille_output = ""
                st.session_state.text2braille_input = ""
                st.rerun()
        st.caption(f"Translation provided by: `{BRAILLE_API_URL}`")

    # --- NEW: Voice Exam Taker Page ---
    elif st.session_state.current_page == "Voice Exam Taker":
        if ve_helpers is None:
            st.error("Voice Exam Taker module failed to load. Cannot display this feature.")
        elif not SPEECH_RECOGNITION_AVAILABLE:
             st.error("Voice Exam Taker requires SpeechRecognition libraries. Cannot display this feature.")
        else:
            # The render function handles its own title and internal logic
            ve_helpers.render_voice_exam_page(app_config) # Pass the main config


    # --- Fallback Page ---
    else:
        st.error("Selected page not found or corresponding modules failed.")
        log.error(f"Reached fallback page renderer. Current page: {st.session_state.get('current_page', 'N/A')}")
        st.warning(f"Available pages: {page_options}")