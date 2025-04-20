import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
SAMPLE_RATE = 16000  # Sample rate for audio recordings
DURATION = 5         # Duration of recordings in seconds
N_MELS = 128         # Number of Mel bands to generate
N_FFT = 2048         # Window size for FFT
HOP_LENGTH = 512     # Hop length for FFT
PREDEFINED_PHRASE = "My voice is my passport, verify me." # Crucial!

REGISTERED_VOICES_DIR = "registered_voices"
TEMP_AUDIO_DIR = "temp_audio"
SIMILARITY_THRESHOLD = 0.60 # IMPORTANT: This needs careful tuning!

# --- Ensure directories exist ---
os.makedirs(REGISTERED_VOICES_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- Core Functions ---

def record_audio(filename, duration=DURATION, sr=SAMPLE_RATE):
    """Records audio from the microphone and saves it to a file."""
    print(f"Please speak the phrase: '{PREDEFINED_PHRASE}'")
    print("Recording...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, sr)
    print(f"Recording saved to {filename}")
    return filename

def extract_mel_spectrogram(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Extracts a Log-Mel Spectrogram from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        # Optional: Trim silence
        y, index = librosa.effects.trim(y, top_db=25) # Adjust top_db as needed

        if len(y) == 0:
            print(f"Warning: Audio file {audio_path} appears to be silent after trimming.")
            return None

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # --- Padding/Truncating to fixed length (Simple Approach) ---
        # We need a fixed length for direct comparison using cosine similarity.
        # Let's define a target length based on duration (adjust if needed)
        target_len = int(np.ceil(DURATION * sr / hop_length)) + 1 # Approximate expected length

        if log_mel_spec.shape[1] < target_len:
            pad_width = target_len - log_mel_spec.shape[1]
            # Pad with the minimum value (silence)
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='minimum')
        else:
            log_mel_spec = log_mel_spec[:, :target_len]

        return log_mel_spec

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def plot_spectrogram(spec, title="Mel Spectrogram"):
    """Utility to plot a spectrogram."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def register_user(user_id):
    """Registers a new user by recording their voice and saving the spectrogram."""
    print(f"\n--- Registering User: {user_id} ---")
    reg_audio_path = os.path.join(TEMP_AUDIO_DIR, f"reg_{user_id}_{int(time.time())}.wav")
    record_audio(reg_audio_path)

    mel_spec = extract_mel_spectrogram(reg_audio_path)

    if mel_spec is not None:
        save_path = os.path.join(REGISTERED_VOICES_DIR, f"{user_id}.npy")
        np.save(save_path, mel_spec)
        print(f"User {user_id} registered successfully. Spectrogram saved.")
        # Optional: Plot the registered spectrogram
        # plot_spectrogram(mel_spec, title=f"Registered Spectrogram - {user_id}")
        os.remove(reg_audio_path) # Clean up temp file
        return True
    else:
        print(f"Registration failed for {user_id}. Could not extract features.")
        if os.path.exists(reg_audio_path):
             os.remove(reg_audio_path)
        return False

def load_registered_voices():
    """Loads all registered voice spectrograms from the directory."""
    registered_data = {}
    print("\nLoading registered voices...")
    for filename in os.listdir(REGISTERED_VOICES_DIR):
        if filename.endswith(".npy"):
            user_id = os.path.splitext(filename)[0]
            filepath = os.path.join(REGISTERED_VOICES_DIR, filename)
            try:
                mel_spec = np.load(filepath)
                registered_data[user_id] = mel_spec
                print(f" - Loaded voice for user: {user_id}")
            except Exception as e:
                print(f"Error loading voice for {user_id} from {filepath}: {e}")
    if not registered_data:
        print("No registered voices found.")
    return registered_data

def verify_user(registered_data):
    """Verifies a user by recording their voice and comparing it to registered data."""
    print("\n--- Voice Verification ---")
    if not registered_data:
        print("No registered users to verify against. Please register users first.")
        return False, None

    verify_audio_path = os.path.join(TEMP_AUDIO_DIR, f"verify_{int(time.time())}.wav")
    record_audio(verify_audio_path)

    verify_spec = extract_mel_spectrogram(verify_audio_path)

    if verify_spec is None:
        print("Verification failed. Could not extract features from the attempt.")
        if os.path.exists(verify_audio_path):
             os.remove(verify_audio_path)
        return False, None

    # Optional: Plot verification spectrogram
    # plot_spectrogram(verify_spec, title="Verification Attempt Spectrogram")

    best_match_user = None
    highest_similarity = -1 # Cosine similarity ranges from -1 to 1

    # Flatten the spectrograms for cosine similarity calculation
    verify_spec_flat = verify_spec.flatten().reshape(1, -1) # Reshape for sklearn

    for user_id, reg_spec in registered_data.items():
        if reg_spec.shape != verify_spec.shape:
            print(f"Warning: Shape mismatch between verification attempt and user {user_id}. Skipping comparison.")
            print(f"  Verification shape: {verify_spec.shape}, Registered shape: {reg_spec.shape}")
            # This should ideally not happen if padding/truncating works correctly.
            continue

        reg_spec_flat = reg_spec.flatten().reshape(1, -1)

        # Calculate Cosine Similarity
        similarity = cosine_similarity(verify_spec_flat, reg_spec_flat)[0][0] # Get the scalar value
        print(f" - Similarity with {user_id}: {similarity:.4f}")

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_user = user_id

    # Clean up temp file
    if os.path.exists(verify_audio_path):
        os.remove(verify_audio_path)

    # --- Decision Logic ---
    if highest_similarity >= SIMILARITY_THRESHOLD:
        print(f"\nVerification Successful! Best match: {best_match_user} (Similarity: {highest_similarity:.4f})")
        print("Access Granted.")
        return True, best_match_user
    else:
        print(f"\nVerification Failed. Highest similarity ({highest_similarity:.4f}) is below threshold ({SIMILARITY_THRESHOLD}).")
        print("Access Denied.")
        return False, None

# --- Main Execution ---
if __name__ == "__main__":
    registered_voices = load_registered_voices()

    while True:
        print("\n--- Voice Authentication System ---")
        print("1. Register New User")
        print("2. Verify Voice (Simulate Exam Entry)")
        print("3. List Registered Users")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            user_id = input("Enter User ID for registration: ")
            if user_id in registered_voices:
                print(f"User ID '{user_id}' already exists. Choose a different ID.")
            elif not user_id.strip():
                 print("User ID cannot be empty.")
            else:
                if register_user(user_id):
                    # Reload registered voices after successful registration
                    registered_voices = load_registered_voices()
        elif choice == '2':
            verify_user(registered_voices)
        elif choice == '3':
            print("\nRegistered Users:")
            if registered_voices:
                for user_id in registered_voices.keys():
                    print(f"- {user_id}")
            else:
                print("None")
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")