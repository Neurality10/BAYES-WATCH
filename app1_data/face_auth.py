import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# --- Configuration ---
REGISTERED_EMBEDDINGS_DIR = "registered_face_embeddings"
# Model for face recognition. Options include:
# "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
# "VGG-Face" or "Facenet512" are good starting points.
MODEL_NAME = "Facenet512"
# Distance metric for comparison. Options: 'cosine', 'euclidean', 'euclidean_l2'
# Using cosine similarity (higher is better) calculated manually for thresholding
DISTANCE_METRIC = 'cosine' # DeepFace internal default, but we calculate manually

# Threshold for cosine similarity (higher means more similar). NEEDS CAREFUL TUNING!
# Start maybe around 0.70-0.75 for Facenet512/VGG-Face, but depends heavily on the model.
# Lower values (e.g., 0.30 for cosine distance) are stricter if using DeepFace.verify directly.
SIMILARITY_THRESHOLD = 0.70

# OpenCV display settings
FRAME_THICKNESS = 2
FONT_SCALE = 0.6
FONT_COLOR_MATCH = (0, 255, 0) # Green
FONT_COLOR_VERIFYING = (255, 255, 0) # Cyan
FONT_COLOR_UNKNOWN = (0, 0, 255) # Red
VERIFICATION_DELAY = 2 # Seconds of consistent match needed

# --- Suppress TensorFlow/DeepFace warnings (optional) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress some TF info/warning messages
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='deepface')


# --- Ensure directory exists ---
os.makedirs(REGISTERED_EMBEDDINGS_DIR, exist_ok=True)

# --- Helper Functions ---

def load_registered_embeddings():
    """Loads known face embeddings and their names from the directory."""
    known_face_embeddings = []
    known_face_names = []
    print("\nLoading registered face embeddings...")
    if not os.path.exists(REGISTERED_EMBEDDINGS_DIR):
        print(f"Directory not found: {REGISTERED_EMBEDDINGS_DIR}")
        return known_face_embeddings, known_face_names

    for filename in os.listdir(REGISTERED_EMBEDDINGS_DIR):
        if filename.endswith(".npy"):
            filepath = os.path.join(REGISTERED_EMBEDDINGS_DIR, filename)
            user_id = os.path.splitext(filename)[0]
            try:
                embedding = np.load(filepath)
                # Ensure embedding is loaded as a 1D array or reshape if needed
                if embedding.ndim > 1:
                     embedding = embedding.flatten() # Ensure it's flat for consistent comparison
                known_face_embeddings.append(embedding)
                known_face_names.append(user_id)
                print(f" - Loaded embedding for user: {user_id}")
            except Exception as e:
                print(f"Error loading embedding for {user_id} from {filepath}: {e}")

    if not known_face_names:
        print("No registered faces found.")
    return known_face_embeddings, known_face_names

def get_face_embedding(frame):
    """Extracts face embedding from a single frame using DeepFace."""
    try:
        # DeepFace expects BGR image (OpenCV default)
        # represent function returns a list of dictionaries, one for each detected face
        embedding_objs = DeepFace.represent(img_path = frame,
                                            model_name = MODEL_NAME,
                                            enforce_detection = True, # Ensure a face is found
                                            detector_backend = 'opencv') # Faster detector

        if len(embedding_objs) > 1:
            print("Warning: Multiple faces detected in frame. Using the first one.")
            # Optionally: could choose the largest face or center face instead

        if len(embedding_objs) > 0:
             # Extract the embedding vector
             embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
             facial_area = embedding_objs[0]["facial_area"] # Get bounding box: {'x': int, 'y': int, 'w': int, 'h': int}
             return embedding, facial_area
        else:
            return None, None # No face detected

    except ValueError as e:
        # Handle cases where DeepFace.represent fails (e.g., no face detected)
        # print(f"DeepFace Error: {e}") # Can be noisy
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during embedding extraction: {e}")
        return None, None


def register_user_face(user_id):
    """Registers a new user by capturing their face and saving the embedding."""
    print(f"\n--- Registering Face Embedding for User: {user_id} ---")
    save_path = os.path.join(REGISTERED_EMBEDDINGS_DIR, f"{user_id}.npy")

    if os.path.exists(save_path):
        print(f"User {user_id} already has a registered face. Overwrite? (y/n)")
        if input().lower() != 'y':
            print("Registration cancelled.")
            return False

    video_capture = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return False

    print("Please look directly at the camera.")
    print("Ensure your face is clearly visible and well-lit.")
    print("Press 's' to save when ready, 'q' to quit.")

    face_embedding = None
    last_message_time = 0
    message = "Initializing..."

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        display_frame = frame.copy()
        temp_embedding, facial_area = get_face_embedding(frame) # Pass the whole frame

        current_time = time.time()
        if temp_embedding is not None:
             if current_time - last_message_time > 1.0: # Update message less frequently
                 message = "Face detected. Ready to save."
                 last_message_time = current_time
             face_embedding = temp_embedding # Store the latest good embedding
             # Draw rectangle from facial_area
             x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
             cv2.rectangle(display_frame, (x, y), (x+w, y+h), FONT_COLOR_MATCH, FRAME_THICKNESS)
        else:
             if current_time - last_message_time > 1.0:
                 message = "No face detected or error."
                 last_message_time = current_time
             face_embedding = None # Reset if no face found

        # Display status message
        cv2.putText(display_frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FONT_COLOR_UNKNOWN, 2)
        cv2.putText(display_frame, "Press 's' to save, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Video - Registration', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if face_embedding is not None:
                np.save(save_path, face_embedding)
                print(f"\nFace embedding for user {user_id} saved successfully.")
                break # Exit registration loop
            else:
                print("Cannot save. Ensure exactly one face is clearly detected.")
                message = "Save failed: No face detected."
                last_message_time = time.time()
        elif key == ord('q'):
            print("Registration cancelled by user.")
            video_capture.release()
            cv2.destroyAllWindows()
            return False

    video_capture.release()
    cv2.destroyAllWindows()
    return True

def verify_face(known_embeddings, known_names):
    """Verifies a face against the known registered face embeddings."""
    print("\n--- Face Verification ---")
    if not known_embeddings:
        print("No registered faces to verify against. Please register users first.")
        return False, None

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return False, None

    print("Look at the camera for verification...")
    print(f"Model: {MODEL_NAME}, Threshold (Cosine Sim): {SIMILARITY_THRESHOLD}")
    print("Press 'q' to quit.")

    last_match_time = 0
    verified_user = None
    access_granted = False
    match_status = "Searching..."

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        display_frame = frame.copy()
        live_embedding, facial_area = get_face_embedding(frame)

        best_match_name = "Unknown"
        best_similarity = -1 # Cosine similarity is between -1 and 1

        if live_embedding is not None:
            # Reshape live embedding for cosine_similarity (expects 2D arrays)
            live_embedding_reshaped = live_embedding.reshape(1, -1)

            # Compare live embedding to all known embeddings
            if known_embeddings:
                 # Calculate cosine similarity
                 similarities = cosine_similarity(live_embedding_reshaped, np.array(known_embeddings))
                 best_match_index = np.argmax(similarities[0])
                 best_similarity = similarities[0][best_match_index]

                 # Check if the best match meets the threshold
                 if best_similarity >= SIMILARITY_THRESHOLD:
                     best_match_name = known_names[best_match_index]
                 # else: Keep best_match_name as "Unknown"

            # Draw bounding box from detection
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            box_color = FONT_COLOR_UNKNOWN
            label = f"Unknown (Sim: {best_similarity:.2f})"

            # --- Authentication Logic ---
            if best_match_name != "Unknown":
                if not access_granted: # First time seeing a potential match
                    match_status = f"Verifying: {best_match_name} ({best_similarity:.2f})"
                    box_color = FONT_COLOR_VERIFYING
                    label = match_status
                    last_match_time = time.time()
                    verified_user = best_match_name
                    access_granted = True # Tentatively grant access
                elif verified_user == best_match_name: # Match persists for the same user
                    match_status = f"Verifying: {best_match_name} ({best_similarity:.2f})"
                    box_color = FONT_COLOR_VERIFYING
                    label = match_status
                    # Check if match persisted long enough
                    if time.time() - last_match_time > VERIFICATION_DELAY:
                         print(f"\nVerification Confirmed! User: {verified_user} (Similarity: {best_similarity:.2f})")
                         print("Access Granted.")
                         video_capture.release()
                         cv2.destroyAllWindows()
                         return True, verified_user
                else: # Switched to a different known user? Reset timer.
                     match_status = f"Verifying: {best_match_name} ({best_similarity:.2f})"
                     box_color = FONT_COLOR_VERIFYING
                     label = match_status
                     last_match_time = time.time()
                     verified_user = best_match_name
                     access_granted = True # Still tentative

            else: # No known match above threshold
                match_status = f"Unknown (Best Sim: {best_similarity:.2f})"
                box_color = FONT_COLOR_UNKNOWN
                label = match_status
                if access_granted: print("Match lost...")
                access_granted = False # Reset access grant
                verified_user = None

            # Draw rectangle and label
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, FRAME_THICKNESS)
            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, box_color, 1)

        else: # No face detected in the frame
             match_status = "No face detected"
             if access_granted: print("Match lost...")
             access_granted = False # Reset access grant
             verified_user = None
             cv2.putText(display_frame, match_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FONT_COLOR_UNKNOWN, 2)


        cv2.imshow('Video - Verification', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Verification cancelled by user.")
            access_granted = False # Ensure we return false if quit manually
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if access_granted and verified_user is not None: # Should have returned earlier if successful
        # This case might be hit if loop breaks right after tentative grant
        print("\nVerification window closed before confirmation delay.")
        return False, None
    else:
        print("\nVerification Failed or Cancelled.")
        print("Access Denied.")
        return False, None

# --- Main Execution ---
if __name__ == "__main__":
    # Load known faces at the start
    known_embeddings, known_names = load_registered_embeddings()

    while True:
        print("\n--- DeepFace Authentication System ---")
        print(f"(Using Model: {MODEL_NAME})")
        print("1. Register New User Face")
        print("2. Verify Face (Simulate Exam Entry)")
        print("3. List Registered Users")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            user_id = input("Enter User ID for face registration: ")
            if not user_id.strip():
                 print("User ID cannot be empty.")
            else:
                if register_user_face(user_id):
                    # Reload registered faces after successful registration
                    known_embeddings, known_names = load_registered_embeddings()
        elif choice == '2':
            verify_face(known_embeddings, known_names)
        elif choice == '3':
            print("\nRegistered Face Users:")
            if known_names:
                for user_id in known_names:
                    print(f"- {user_id}")
            else:
                print("None")
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")