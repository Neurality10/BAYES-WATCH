# speech_to_text.py
"""Handles Speech-to-Text initialization and transcription."""

import streamlit as st
import speech_recognition as sr
import io
import traceback
from pydub import AudioSegment
from config import get_config_value

# Cache the recognizer instance for efficiency
@st.cache_resource
def get_recognizer():
    """Initializes and returns a SpeechRecognition Recognizer instance."""
    # --- ADDED DETAILED LOGGING ---
    print("DEBUG STT INIT - ENTER: get_recognizer()")
    try:
        print("DEBUG STT INIT - Step 1: Initializing sr.Recognizer()...")
        r = sr.Recognizer()
        print("DEBUG STT INIT - Step 1: sr.Recognizer() initialized successfully.")

        # Get config values
        pause_threshold = float(get_config_value('VOICE_ENGINE_PAUSE_THRESHOLD', 0.8))
        dynamic_energy = bool(get_config_value('VOICE_ENGINE_DYNAMIC_ENERGY', True))
        energy_threshold = int(get_config_value('VOICE_ENGINE_ENERGY_THRESHOLD', 400))

        print(f"DEBUG STT INIT - Step 2: Applying config: pause={pause_threshold}, dynamic={dynamic_energy}, energy={energy_threshold}")
        r.pause_threshold = pause_threshold
        r.dynamic_energy_threshold = dynamic_energy
        if not dynamic_energy:
             r.energy_threshold = energy_threshold
        print("DEBUG STT INIT - Step 2: Config applied successfully.")

        # Try listing microphones within Streamlit context (optional but informative)
        try:
            print("DEBUG STT INIT - Step 3: Listing microphones via sr.Microphone...")
            mic_names = sr.Microphone.list_microphone_names()
            print(f"DEBUG STT INIT - Step 3: Microphones found ({len(mic_names)}): {mic_names}")
        except Exception as mic_err:
             print(f"DEBUG STT INIT - Step 3: Error listing microphones: {mic_err}")
             # Don't necessarily fail initialization just because listing failed,
             # but it's useful info. The main check is opening the default mic.

        # Try opening default microphone briefly to ensure access
        try:
            print("DEBUG STT INIT - Step 4: Testing default microphone access...")
            with sr.Microphone() as source:
                 print(f"DEBUG STT INIT - Step 4: Default microphone opened (Index: {source.device_index}).")
            print("DEBUG STT INIT - Step 4: Microphone test successful.")
        except Exception as mic_open_err:
             print(f"ERROR STT INIT - Step 4: Failed to open default microphone: {mic_open_err}")
             # This is likely the cause if it gets here
             st.session_state.speech_rec_error = True
             st.session_state.last_feedback = f"Voice recognizer init error: Failed to access microphone ({mic_open_err})"
             print("DEBUG STT INIT - EXIT: get_recognizer() - FAILED (Microphone access error)")
             return None # Fail initialization

        print(f"DEBUG STT INIT - Recognizer fully configured. Dynamic Energy: {r.dynamic_energy_threshold}, "
              f"Energy Threshold: {r.energy_threshold}, Pause Threshold: {r.pause_threshold}")
        st.session_state.speech_rec_error = False # Mark as successful
        print("DEBUG STT INIT - EXIT: get_recognizer() - SUCCESS")
        return r

    except Exception as e:
        # Catch any other unexpected error during initialization
        st.error(f"Fatal Error: Failed to initialize voice recognizer: {e}", icon="🎙️")
        print(f"EXCEPTION STT INIT: SR Recognizer initialization failed unexpectedly.\n{traceback.format_exc()}")
        st.session_state.last_feedback = f"Voice recognizer init error: {e}"
        st.session_state.speech_rec_error = True
        print("DEBUG STT INIT - EXIT: get_recognizer() - FAILED (Unexpected Exception)")
        return None

# --- transcribe_audio function remains the same ---
def transcribe_audio(audio_info, recognizer):
    """
    Transcribes audio bytes using the provided recognizer.
    audio_info is the dictionary returned by streamlit-mic-recorder
    (expects keys like 'bytes', 'sample_rate', 'sample_width').
    Returns the transcribed text (str) or None if transcription fails.
    """
    if not recognizer or st.session_state.get('speech_rec_error', True):
        # Don't show redundant error if init failed
        # st.error("Voice recognizer is not available.")
        if st.session_state.get('speech_rec_error', True):
             print("ERROR STT: transcribe_audio called but recognizer initialization failed earlier.")
        else:
             print("ERROR STT: transcribe_audio called but recognizer is None.")
             st.session_state.last_feedback = "Recognizer unavailable." # Generic feedback
        return None

    if not audio_info or 'bytes' not in audio_info or not audio_info['bytes']:
         st.warning("No audio data received from recorder.")
         st.session_state.last_feedback = "No audio captured from mic."
         print("WARNING STT: transcribe_audio called with no audio bytes.")
         return None

    print("DEBUG STT: Starting audio transcription...")
    st.session_state.last_feedback = "Processing recorded audio..."

    try:
        raw_bytes = audio_info['bytes']
        audio_io = io.BytesIO(raw_bytes)
        print(f"DEBUG STT: Received {len(raw_bytes)} bytes.")

        try:
            print("DEBUG STT: Attempting pydub load...")
            sound = AudioSegment.from_file(audio_io)
            print(f"DEBUG STT: pydub loaded audio successfully. Frame Rate: {sound.frame_rate}, Channels: {sound.channels}")
        except Exception as pydub_err:
            print(f"ERROR STT: pydub failed to load audio: {pydub_err}. Ensure FFmpeg is installed and in PATH.")
            st.error(f"Error processing audio format: {pydub_err}. Is FFmpeg installed?")
            st.session_state.last_feedback = "Error: Could not process audio format (FFmpeg missing?)."
            return None

        wav_io = io.BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0)
        print("DEBUG STT: Audio exported to WAV format in memory.")

        with sr.AudioFile(wav_io) as source:
            try:
                print("DEBUG STT: Recognizer attempting to record audio data from source...")
                audio_data = recognizer.record(source)
                print("DEBUG STT: Recognizer recorded audio data from WAV.")
            except Exception as record_err:
                 print(f"ERROR STT: recognizer.record() failed: {record_err}")
                 st.error(f"Error during audio recording phase: {record_err}")
                 st.session_state.last_feedback = "Error during STT recording phase."
                 return None

        language_code = get_config_value('VOICE_ENGINE_LANGUAGE', 'en-US')
        print(f"DEBUG STT: Sending audio to Google Speech Recognition (lang={language_code})...")
        text = recognizer.recognize_google(audio_data, language=language_code)
        text = text.strip()

        st.session_state.last_feedback = f"Recognized (STT): \"{text}\""
        print(f"INFO STT: Transcription successful: '{text}'")
        return text

    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
        st.session_state.last_feedback = "Couldn't understand audio (STT)."
        print("WARNING STT: SR UnknownValueError during transcription.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech Recognition service error: {e}")
        st.session_state.last_feedback = f"Speech service error (STT): {e}"
        print(f"ERROR STT: SR RequestError: {e}")
        return None
    except ImportError:
         st.error("Error: Audio processing library (`pydub` or `ffmpeg`) missing or broken.")
         st.session_state.last_feedback = "Audio processing library missing (STT)."
         print("ERROR STT: pydub/ffmpeg import error during transcription.")
         return None
    except Exception as e:
        st.error(f"Error during audio processing or transcription: {e}")
        st.session_state.last_feedback = f"Transcription error (STT): {e}"
        print(f"ERROR STT: Unexpected error during transcription: {e}")
        traceback.print_exc()
        return None