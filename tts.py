# tts.py
"""
Generates audio files from transcript text segments using Google Text-to-Speech (gTTS).
Requires an active internet connection.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Attempt to import gTTS and handle potential errors
try:
    from gtts import gTTS, gTTSError
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    # Logged later if TTS is attempted without the library

# Setup logger for this module
log = logging.getLogger(__name__)

# --- Main Orchestrator Entry Point ---
def generate_audio_files(transcript_json_path: Path, output_audio_dir: Path, config: Dict[str, Any]) -> bool:
    """
    Main function called by orchestrator. Reads transcripts from JSON, generates MP3
    audio files using gTTS, and saves them to the specified directory.

    Args:
        transcript_json_path: Path to the JSON file containing transcript data.
        output_audio_dir: Path to the directory where generated MP3 files should be saved.
        config: The pipeline configuration dictionary.

    Returns:
        True if the process ran and attempted generation (even with errors),
        False if setup failed (e.g., gTTS not installed, input not found).
    """
    start_time = time.time()
    log.info(f"Starting TTS generation using gTTS for: {transcript_json_path.name}")

    # --- Prerequisite Checks ---
    if not GTTS_AVAILABLE:
        log.error("gTTS library not found. Cannot generate speech. Please install it: pip install gTTS")
        return False

    if not transcript_json_path.is_file():
        log.error(f"Transcript JSON file not found: {transcript_json_path}")
        return False

    # --- Get Config ---
    language = config.get('TTS_LANGUAGE', 'en')
    # Optional: Add a small delay between generations (good practice for online APIs)
    delay_between_generations = config.get('TIMEOUTS', {}).get('tts_delay', 1.0) # Default 1 sec delay

    # --- Setup ---
    log.info(f"Using language: {language}")
    log.info("NOTE: gTTS requires an active internet connection.")

    try:
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output audio files will be saved in: {output_audio_dir}")
    except Exception as e:
        log.exception(f"Failed to create output directory {output_audio_dir}: {e}")
        return False

    # --- Load Transcripts ---
    try:
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            transcripts_data = json.load(f)
        if not isinstance(transcripts_data, list):
            log.error(f"Expected a list in transcript JSON, but got {type(transcripts_data)}. Cannot process.")
            return False
        log.info(f"Loaded {len(transcripts_data)} transcript items from {transcript_json_path.name}")
    except json.JSONDecodeError:
        log.exception(f"Error decoding JSON from {transcript_json_path}. Please check the file format.")
        return False
    except Exception as e:
        log.exception(f"An unexpected error occurred while reading the transcript JSON file: {e}")
        return False

    if not transcripts_data:
        log.warning("Transcript JSON file is empty. No audio files to generate.")
        return True # Process completed successfully (nothing to do)

    # --- Process Transcripts using gTTS ---
    log.info("--- Starting Speech Generation ---")
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    total_items = len(transcripts_data)

    for i, item in enumerate(transcripts_data):
        if not isinstance(item, dict):
            log.warning(f"Skipping item {i+1}: Expected a dictionary, got {type(item)}")
            skipped_count += 1
            continue

        # Use the 'id' or 'filename_base' for naming, ensure uniqueness
        file_id = item.get("id")
        filename_base = item.get("filename_base")
        if filename_base:
            output_name = filename_base
        elif file_id is not None:
             output_name = f"transcript_{file_id:03d}" # Use ID if base is missing
        else:
             output_name = f"transcript_{i+1:03d}" # Fallback to index
             log.warning(f"Item {i+1} missing 'id' and 'filename_base', using index for filename.")

        transcript_text = item.get("transcript", "").strip()

        log.info(f"Processing Item {i+1}/{total_items} (ID: {file_id}, Base: {filename_base})...")

        if not transcript_text:
            log.warning(f"  Skipping '{output_name}': Empty transcript text.")
            skipped_count += 1
            continue

        # gTTS saves as MP3
        output_path = output_audio_dir / f"{output_name}.mp3"
        log.debug(f"  Target output file: {output_path}")

        # --- Optional: Skip if file exists ---
        # Uncomment the following block if you want to skip regeneration
        # if output_path.exists():
        #     log.info(f"  Skipping '{output_name}': Output file already exists.")
        #     skipped_count += 1
        #     continue # Count as skipped, not success/fail if skipping existing

        log.debug(f"  Generating speech for: \"{transcript_text[:80]}{'...' if len(transcript_text) > 80 else ''}\"")
        generation_start_time = time.time()
        try:
            # <<< gTTS Call >>>
            tts_instance = gTTS(text=transcript_text, lang=language, slow=False)
            tts_instance.save(str(output_path)) # gTTS save expects string path
            # <<< End gTTS Call >>>

            generation_time = time.time() - generation_start_time
            log.info(f"  Speech generated and saved to '{output_path.name}' in {generation_time:.2f}s.")
            successful_count += 1

            # Add delay
            if delay_between_generations > 0 and i < total_items - 1: # Don't delay after last item
                log.debug(f"  Waiting for {delay_between_generations:.1f}s...")
                time.sleep(delay_between_generations)

        except gTTSError as e:
             log.error(f"  gTTS Error for '{output_name}': {e}. Check internet connection, language code ('{language}'), or potential rate limits.")
             failed_count += 1
             # Optional: Add a longer delay after an error?
             # time.sleep(5)
             # break # Uncomment to stop pipeline on first gTTS error
        except Exception as e:
            log.exception(f"  Unexpected Error for '{output_name}': {e}")
            failed_count += 1
            # break # Uncomment to stop pipeline on first general error

    # --- Final Summary ---
    log.info("--- TTS Speech Generation Finished ---")
    log.info(f"Summary:")
    log.info(f"  Successfully generated: {successful_count}")
    log.info(f"  Failed generations:   {failed_count}")
    log.info(f"  Skipped (empty/exist): {skipped_count}")
    log.info(f"  Total items attempted: {total_items}")
    log.info(f"Audio files saved in: {output_audio_dir} (as MP3)")
    elapsed_total = time.time() - start_time
    log.info(f"Total TTS generation time: {elapsed_total:.2f}s")

    # Return True if the process ran, False only if initial setup failed.
    # Individual gTTS errors don't cause a False return unless 'break' is uncommented above.
    return True

# No `if __name__ == "__main__":` block needed.