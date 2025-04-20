import pygame
import os
import time
import sys

# --- Configuration ---
# Using raw strings (r"...") for Windows paths is good practice
LETTER_AUDIO_DIR = r"C:\Users\HARSH\Desktop\hack36\audio\letters_mp3"

# --- NEW: Define paths for the three pitch levels ---
# Get the directory and base filename from the user's provided path
BLEEP_BASE_PATH = r"C:\Users\HARSH\Desktop\hack36\beep.mp3"
bleep_dir = os.path.dirname(BLEEP_BASE_PATH)
bleep_filename_base = os.path.splitext(os.path.basename(BLEEP_BASE_PATH))[0] # e.g., "beep"
bleep_extension = os.path.splitext(BLEEP_BASE_PATH)[1] # e.g., ".mp3"

BLEEP_SOUND_HIGH = os.path.join(bleep_dir, f"{bleep_filename_base}_high{bleep_extension}")
BLEEP_SOUND_MID = BLEEP_BASE_PATH # The original file is the middle pitch
BLEEP_SOUND_LOW = os.path.join(bleep_dir, f"{bleep_filename_base}_low{bleep_extension}")

# Store paths for easier checking
bleep_files = {
    "high": BLEEP_SOUND_HIGH,
    "mid": BLEEP_SOUND_MID,
    "low": BLEEP_SOUND_LOW
}

# Check existence of all required bleep files
found_all_bleeps = True
for key, path in bleep_files.items():
    if not os.path.exists(path):
        print(f"Error: Bleep sound file for '{key}' pitch not found at: {path}")
        found_all_bleeps = False

if not found_all_bleeps:
    print("\nPlease create the high, mid (original), and low pitch bleep sound files.")
    print(f"(Expected names like: {os.path.basename(BLEEP_SOUND_HIGH)}, {os.path.basename(BLEEP_SOUND_MID)}, {os.path.basename(BLEEP_SOUND_LOW)})")
    sys.exit(1)
# --- End New Bleep Config ---

# Check if letter audio directory exists
if not os.path.isdir(LETTER_AUDIO_DIR):
     print(f"Error: Letter audio directory '{LETTER_AUDIO_DIR}' not found.")
     print("Please create this folder and put your A.mp3, B.mp3... files inside.")
     sys.exit(1)

# Panning values: (left_volume, right_volume)
LEFT_PAN = (0.9, 0.1)
RIGHT_PAN = (0.1, 0.9)

# Delay between bleeps in seconds
BLEEP_DELAY = 0.01 # Time AFTER the bleep sound finishes
INTER_LETTER_DELAY = 0.1 # Pause after letter name

# --- Braille Patterns (Unchanged) ---
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

# --- Pygame Initialization (Unchanged) ---
try:
    pygame.mixer.pre_init(44100, -16, 2, 1024)
    pygame.init()
    pygame.mixer.init()
    print("Pygame initialized successfully.")
except pygame.error as e:
    print(f"Fatal Error: Could not initialize Pygame or Pygame Mixer: {e}")
    sys.exit(1)

# --- Load Bleep Sounds (NEW: Load all three) ---
bleep_sounds = {}
bleep_durations_ms = {}
bleep_delay_ms = int(BLEEP_DELAY * 500)
try:
    print("Loading bleep sounds...")
    for key, path in bleep_files.items():
         print(f"  Loading {key}: {path}")
         bleep_sounds[key] = pygame.mixer.Sound(path)
         bleep_durations_ms[key] = int(bleep_sounds[key].get_length() * 1000)
    print("High, Mid, and Low bleep sounds loaded.")
except pygame.error as e:
    print(f"Fatal Error: Could not load one of the bleep sounds: {e}")
    sys.exit(1)
# Use average or mid duration for consistent delay after sound
average_bleep_duration_ms = bleep_durations_ms["mid"] # Or calculate average


# --- Helper Function to Play Audio (MODIFIED) ---
def play_braille_audio(letter):
    """Plays letter name, then Braille bleeps using pitch for rows and pan for columns."""
    if letter not in BRAILLE_PATTERNS:
        print(f"Warning: Braille pattern for '{letter}' not defined.")
        return

    print(f"\n--- Letter: {letter} ---")
    pygame.mixer.music.stop()
    pygame.mixer.stop()
    pygame.time.wait(50)

    # 1. Play Letter Name (Unchanged)
    letter_file_path = os.path.join(LETTER_AUDIO_DIR, f"{letter}.mp3")
    if os.path.exists(letter_file_path):
        try:
            print(f"Playing name: {letter_file_path}")
            pygame.mixer.music.load(letter_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.event.pump()
                pygame.time.Clock().tick(15)
        except pygame.error as e:
            print(f"Error playing {letter_file_path}: {e}")
    else:
        print(f"Warning: Letter audio file not found: {letter_file_path}")
        pygame.time.wait(100)

    pygame.time.wait(int(INTER_LETTER_DELAY * 1000))

    # 2. Play Braille Bleeps (MODIFIED: Select sound based on row, pan based on column)
    pattern = BRAILLE_PATTERNS[letter]
    print(f"Braille Pattern: {pattern}")
    print("Playing bleeps (Pitch: High/Mid/Low | Pan: Left/Right):")

    dot_sequence = [1, 2, 3, 4, 5, 6] # Check dots in standard order

    for dot_number in dot_sequence:
        dot_index = dot_number - 1
        if pattern[dot_index] == 1: # Is this dot raised?
            channel = pygame.mixer.find_channel(True) # Find next available channel, force=True

            if channel:
                # Determine Panning (Left/Right Column)
                if dot_number <= 3:
                    pan = LEFT_PAN
                    col_name = "Left"
                else:
                    pan = RIGHT_PAN
                    col_name = "Right"

                # Determine Pitch (Top/Mid/Bottom Row) & Select Sound
                current_bleep_sound = None
                row_name = ""
                if dot_number in [1, 4]: # Top row
                    current_bleep_sound = bleep_sounds["high"]
                    row_name = "High"
                elif dot_number in [2, 5]: # Middle row
                    current_bleep_sound = bleep_sounds["mid"]
                    row_name = "Mid"
                elif dot_number in [3, 6]: # Bottom row
                    current_bleep_sound = bleep_sounds["low"]
                    row_name = "Low"

                # Play the selected sound with the calculated panning
                if current_bleep_sound:
                    print(f"  Dot {dot_number}: {row_name} / {col_name}")
                    channel.set_volume(pan[0], pan[1]) # Set stereo panning
                    channel.play(current_bleep_sound)  # Play correct pitch

                    # Wait using a consistent duration + delay
                    total_wait_ms = average_bleep_duration_ms + bleep_delay_ms
                    pygame.time.wait(total_wait_ms)
                else:
                     # Should not happen if dot_number is 1-6, but safety check
                     print(f"  Dot {dot_number}: Error determining row sound.")
                     pygame.time.wait(bleep_delay_ms) # Wait delay anyway

            else:
                print(f"Warning: Could not find available audio channel for dot {dot_number}.")
                pygame.time.wait(bleep_delay_ms) # Wait delay anyway

    print("--- Finished Letter ---")


# --- Main Loop (Unchanged from your provided version) ---
current_letter_index = 0
screen_width = 300
screen_height = 300
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Braille Audio Learner (Pitch)")
try:
    main_font = pygame.font.Font(None, 72)
except Exception:
    main_font = pygame.font.SysFont("arial", 60)

print("\n--- Braille Audio Learner (Pitch Enhanced) ---")
print("Controls:")
print("  RIGHT Arrow: Next Letter")
print("  LEFT Arrow:  Previous Letter")
print("  SPACEBAR:    Repeat Current Letter")
print("  ESCAPE:      Quit")
print("\nEnsure high, mid (original), and low pitch bleep sounds exist.")
print(f"(e.g., {os.path.basename(BLEEP_SOUND_HIGH)}, {os.path.basename(BLEEP_SOUND_MID)}, {os.path.basename(BLEEP_SOUND_LOW)})")
print("Starting with 'A'. Press Space or Right Arrow to begin.")

running = True
needs_initial_play = True # Maybe set to False if you want to require first key press

while running:
    play_sound_now = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            needs_initial_play = False
            if event.key == pygame.K_RIGHT:
                current_letter_index = (current_letter_index + 1) % len(ALPHABET)
                play_sound_now = True
            elif event.key == pygame.K_LEFT:
                current_letter_index = (current_letter_index - 1) % len(ALPHABET)
                play_sound_now = True
            elif event.key == pygame.K_SPACE:
                 play_sound_now = True
            elif event.key == pygame.K_ESCAPE:
                running = False

    if play_sound_now:
        current_letter = ALPHABET[current_letter_index]
        play_braille_audio(current_letter)

    screen.fill((20, 20, 50))
    current_letter_display = ALPHABET[current_letter_index]
    text_surface = main_font.render(current_letter_display, True, (220, 220, 255))
    text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()

    pygame.time.Clock().tick(30)

# --- Cleanup ---
print("\nExiting Braille Learner.")
pygame.mixer.quit()
pygame.quit()
sys.exit()