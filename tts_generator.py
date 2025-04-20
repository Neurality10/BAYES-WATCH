# --- START OF FILE tts_generator.py ---

import os
import re
# import markdown # No longer needed if parsing directly
# from bs4 import BeautifulSoup # No longer needed if parsing directly
from gtts import gTTS
import pyttsx3
import time
import urllib.request
import requests
from pathlib import Path # Use pathlib for better path handling

# --- Proxy Functions (Remain Unchanged) ---
def get_system_proxies():
    """
    Get system proxy settings from environment variables or system configuration.

    Returns:
        dict: Dictionary containing proxy settings for http and https
    """
    proxies = {}
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    if not proxies:
        try:
            system_proxies = urllib.request.getproxies()
            if system_proxies:
                proxies = system_proxies
        except Exception as e:
            print(f"Error detecting system proxies: {e}")
    return proxies

def test_proxy(proxies):
    """ Test proxy connection """
    if not proxies: return False
    try:
        response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Proxy test failed: {e}")
        return False

# --- TTSGenerator Class ---
class TTSGenerator:
    """
    Text-to-Speech generator that converts markdown content to audio files.
    Supports multiple TTS engines and handles document structure including image descriptions.
    """
    def __init__(self, engine_type="gtts", voice_id=None, rate=150, volume=1.0, language='en'):
        """ Initialize the TTS generator """
        self.engine_type = engine_type.lower()
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.language = language
        self.engine = None
        self.proxies = get_system_proxies() # Proxy detection remains

        if self.proxies:
            print(f"Detected proxy settings: {self.proxies}")
            # Proxy test remains, but will only affect gTTS if used
            if test_proxy(self.proxies):
                print("Proxy test successful, will use proxy for gTTS if needed")
            else:
                print("Proxy test failed, gTTS will use direct connection (or fail)")
                # We don't clear self.proxies here, gTTS might still attempt with it
        else:
            print("No proxy detected, gTTS will use direct connection")

        # Initialize the offline engine if chosen or needed as fallback
        if self.engine_type == "pyttsx3":
            try:
                self.engine = pyttsx3.init()
                if self.voice_id: self.engine.setProperty('voice', self.voice_id)
                self.engine.setProperty('rate', self.rate)
                self.engine.setProperty('volume', self.volume)
                self.available_voices = self.engine.getProperty('voices')
            except Exception as e_init:
                 print(f"ERROR initializing pyttsx3: {e_init}. pyttsx3 might not work.")
                 self.engine = None # Ensure engine is None if init fails

    # --- get_available_voices (Remains Unchanged) ---
    def get_available_voices(self):
        # ... (code remains the same) ...
        if self.engine_type == "pyttsx3" and self.engine:
             return [{"id": voice.id, "name": voice.name, "languages": voice.languages}
                    for voice in self.available_voices]
        elif self.engine_type == "gtts":
             return [{"id": "default", "name": "Google TTS Default", "languages": [self.language]}]
        return []


    # --- _clean_text (Remains Mostly Unchanged) ---
    # Added removal of potential markdown image syntax leftover
    def _clean_text(self, text):
        """ Clean and prepare text for TTS processing """
        # Remove specific markdown syntax first
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)    # Italic
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)      # Inline code
        cleaned = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', cleaned) # Image syntax (keep alt text)
        cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned)  # Links (keep link text)

        # Remove potentially problematic characters for TTS
        cleaned = cleaned.replace('#', '') # Remove heading markers if they slip through

        # Replace HTML entities (might still appear if source was complex)
        cleaned = cleaned.replace('&nbsp;', ' ')
        cleaned = cleaned.replace('&lt;', 'less than')
        cleaned = cleaned.replace('&gt;', 'greater than')
        cleaned = cleaned.replace('&amp;', 'and')

        # Standardize whitespace and add pauses (careful not to add too many)
        cleaned = ' '.join(cleaned.split()) # Normalize whitespace
        cleaned = cleaned.replace('. ', '. ... ') # Add pause after sentences
        cleaned = cleaned.replace('! ', '! ... ')
        cleaned = cleaned.replace('? ', '? ... ')
        cleaned = cleaned.replace(': ', ': .. ')
        cleaned = cleaned.replace('; ', '; .. ')
        cleaned = cleaned.replace('\n', ' ') # Replace newlines with space

        return cleaned.strip()

    # --- >>> REVISED Markdown Parsing <<< ---
    def _extract_content_from_markdown(self, md_file_path):
        """
        Extract structured content directly from the book-like markdown file
        using regular expressions.
        """
        content_sections = []
        paragraph_buffer = []

        # Regex patterns
        # Match headings (e.g., # Title, ## Section)
        heading_pattern = re.compile(r'^(#+)\s+(.*)')
        # Match images ![Alt text](path) - Captures Alt text
        image_pattern = re.compile(r'^!\[(.*?)\]\(.*?\)')
         # Match unordered list items (*, -, +) - Captures item text
        ulist_item_pattern = re.compile(r'^\s*[\*\-\+]\s+(.*)')
        # Match ordered list items (1., 2.) - Captures item text
        olist_item_pattern = re.compile(r'^\s*\d+\.\s+(.*)')
        # Match table links [Link to Table X](path) - We'll just announce a table
        table_link_pattern = re.compile(r'^\[Link to Table \d+\]\(.*?\)')

        print(f"DEBUG (TTS): Parsing markdown file: {md_file_path}")
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()

                    # Skip empty lines, but process buffer if needed
                    if not line:
                        if paragraph_buffer:
                            text = self._clean_text(" ".join(paragraph_buffer))
                            if text: content_sections.append({"type": "paragraph", "text": text})
                            paragraph_buffer = []
                        continue

                    heading_match = heading_pattern.match(line)
                    image_match = image_pattern.match(line)
                    ulist_match = ulist_item_pattern.match(line)
                    olist_match = olist_item_pattern.match(line)
                    table_match = table_link_pattern.match(line)

                    # If it's a special element, process the paragraph buffer first
                    is_special = heading_match or image_match or ulist_match or olist_match or table_match
                    if is_special and paragraph_buffer:
                        text = self._clean_text(" ".join(paragraph_buffer))
                        if text: content_sections.append({"type": "paragraph", "text": text})
                        paragraph_buffer = []

                    # Process the matched element
                    if heading_match:
                        level = len(heading_match.group(1))
                        text = self._clean_text(heading_match.group(2))
                        prefix = "Chapter: " if level == 1 else "Section: " if level == 2 else "Subsection: "
                        if text: content_sections.append({"type": f"heading_{level}", "text": f"{prefix}{text}"})
                    elif image_match:
                        alt_text = self._clean_text(image_match.group(1))
                        if alt_text: # Only add if description exists
                            content_sections.append({"type": "image_description", "text": f"Image description: {alt_text}"})
                        else:
                            content_sections.append({"type": "image_description", "text": "An image is present."}) # Announce image even if no alt text
                    elif ulist_match:
                        text = self._clean_text(ulist_match.group(1))
                        if text: content_sections.append({"type": "list_item", "text": f"Bullet point: {text}"})
                    elif olist_match:
                        text = self._clean_text(olist_match.group(1))
                        if text: content_sections.append({"type": "list_item", "text": f"List item: {text}"}) # Generic prefix for ordered
                    elif table_match:
                         content_sections.append({"type": "table_announcement", "text": "A table is referenced here."})
                    elif not is_special:
                        # If nothing matched, it's part of a paragraph
                        paragraph_buffer.append(line)
                    # else: It was a special line that we handled

            # Add any remaining paragraph content at the end of the file
            if paragraph_buffer:
                text = self._clean_text(" ".join(paragraph_buffer))
                if text: content_sections.append({"type": "paragraph", "text": text})

        except Exception as e:
            print(f"ERROR reading or parsing markdown file {md_file_path}: {e}")

        print(f"DEBUG (TTS): Extracted {len(content_sections)} content sections.")
        # Optional: Print extracted sections for debugging
        # for section in content_sections: print(f"  - Type: {section['type']}, Text: {section['text'][:80]}...")
        return content_sections

    # --- generate_speech_pyttsx3 (Remains Unchanged) ---
    def generate_speech_pyttsx3(self, text, output_file):
        """ Generate speech using pyttsx3 """
        if not self.engine:
             print("ERROR: pyttsx3 engine not initialized. Cannot generate speech.")
             # Attempt to initialize now as a last resort
             try:
                 self.engine = pyttsx3.init()
                 if self.voice_id: self.engine.setProperty('voice', self.voice_id)
                 self.engine.setProperty('rate', self.rate)
                 self.engine.setProperty('volume', self.volume)
                 print("Initialized pyttsx3 engine on demand.")
             except Exception as e_init:
                 print(f"ERROR: Failed to initialize pyttsx3 on demand: {e_init}")
                 return # Cannot proceed

        try:
            print(f"DEBUG (TTS): Generating pyttsx3 speech for text (length {len(text)})...")
            # Ensure output path uses .wav for pyttsx3
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.mp3':
                 output_path = output_path.with_suffix('.wav')
                 print(f"DEBUG (TTS): Changed output to WAV for pyttsx3: {output_path.name}")
            self.engine.save_to_file(text, str(output_path)) # Pass path as string
            self.engine.runAndWait()
            print(f"DEBUG (TTS): pyttsx3 speech saved to {output_path.name}")
        except Exception as e:
             print(f"ERROR during pyttsx3 generation: {e}")


    # --- generate_speech_gtts (Remains Mostly Unchanged) ---
    # Added explicit print statements
    def generate_speech_gtts(self, text, output_file):
        """ Generate speech using gTTS """
        try:
            print(f"DEBUG (TTS): Attempting gTTS generation for text (length {len(text)})...")
            tts = gTTS(text=text, lang=self.language, slow=False)
            # Proxy handling remains
            if self.proxies:
                session = requests.Session()
                session.proxies = self.proxies
                # Verify proxy format for requests
                valid_proxies = {}
                for key, value in self.proxies.items():
                    if isinstance(value, str) and ('://' in value or value.startswith('/')):
                         valid_proxies[key] = value
                    else:
                         print(f"WARNING (TTS): Skipping invalid proxy format for requests: {key}={value}")
                if valid_proxies:
                     session.proxies = valid_proxies
                     tts.session = session
                     print(f"DEBUG (TTS): Using requests Session with proxies for gTTS: {valid_proxies}")
                else:
                     print("DEBUG (TTS): No valid proxies found for requests Session.")
            else:
                print("DEBUG (TTS): No proxy detected, using direct connection for gTTS.")

            # Ensure output path uses .mp3 for gTTS
            output_path = Path(output_file)
            if output_path.suffix.lower() != '.mp3':
                output_path = output_path.with_suffix('.mp3')
                print(f"DEBUG (TTS): Ensured output is MP3 for gTTS: {output_path.name}")

            tts.save(str(output_path)) # Pass path as string
            print(f"DEBUG (TTS): gTTS speech saved to {output_path.name}")
            return True
        except requests.exceptions.ProxyError as pe:
             print(f"ERROR (TTS): gTTS ProxyError: {pe}")
             print(" -> Check proxy settings (address, port, authentication if needed).")
             print(" -> Falling back to pyttsx3 (offline TTS)...")
             return False
        except Exception as e:
            print(f"ERROR (TTS): gTTS failed: {type(e).__name__} - {e}")
            print(" -> Check network connection and gTTS compatibility.")
            print(" -> Falling back to pyttsx3 (offline TTS)...")
            return False

    # --- generate_speech (Remains Mostly Unchanged) ---
    # Added print statement for fallback
    def generate_speech(self, text, output_file):
        """ Generate speech using the configured engine """
        if not text:
            print("WARNING (TTS): Received empty text string, skipping speech generation.")
            return

        if self.engine_type == "pyttsx3":
            self.generate_speech_pyttsx3(text, output_file)
        elif self.engine_type == "gtts":
            success = self.generate_speech_gtts(text, output_file)
            if not success:
                print("INFO (TTS): Falling back to pyttsx3 due to gTTS failure.")
                self.generate_speech_pyttsx3(text, output_file) # Call offline fallback

    # --- process_markdown_to_speech (Remains Mostly Unchanged) ---
    # Uses revised _extract_content_from_markdown
    def process_markdown_to_speech(self, md_file_path, output_dir=None, chunk_size=4000):
        """ Process markdown file to speech """
        md_file_path_obj = Path(md_file_path) # Work with Path object
        if not md_file_path_obj.is_file():
            print(f"Error: Markdown file not found: {md_file_path}")
            return []

        # Determine output directory
        output_dir_obj = Path(output_dir) if output_dir else md_file_path_obj.parent
        output_dir_obj.mkdir(parents=True, exist_ok=True) # Create if needed

        base_filename = md_file_path_obj.stem # Get filename without suffix

        # Extract content using the revised method
        content_sections = self._extract_content_from_markdown(md_file_path)
        if not content_sections:
             print(f"WARNING (TTS): No content extracted from {md_file_path}. No audio will be generated.")
             return []

        # Group content into chunks
        chunks = []
        current_chunk = ""
        for section in content_sections:
            section_text = section.get("text", "")
            if not section_text: continue # Skip empty sections

            # Add pauses between different section types for better flow
            if current_chunk and (section["type"].startswith("heading_") or section["type"] == "image_description"):
                 current_chunk += " ... " # Add pause before important elements

            # Check chunk size limit
            if len(current_chunk) + len(section_text) + 1 > chunk_size and current_chunk: # +1 for space
                chunks.append(current_chunk.strip())
                current_chunk = section_text
            else:
                if current_chunk: # Add space only if chunk isn't empty
                    current_chunk += " " + section_text
                else:
                    current_chunk = section_text

        if current_chunk.strip(): # Add the last chunk if it has content
            chunks.append(current_chunk.strip())

        print(f"Split content into {len(chunks)} chunks for TTS processing.")

        # Generate speech for each chunk
        output_files = []
        num_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            # Use engine-appropriate suffix initially
            suffix = ".mp3" if self.engine_type == "gtts" else ".wav"
            # Determine filename (add part number only if multiple chunks)
            filename_base = f"{base_filename}_part{i+1}" if num_chunks > 1 else base_filename
            output_file_path = output_dir_obj / f"{filename_base}{suffix}"

            print(f"\n--- Generating Chunk {i+1}/{num_chunks} ---")
            print(f"Output file: {output_file_path.name}")
            # print(f"DEBUG (TTS): Text for chunk {i+1}:\n'''{chunk[:200]}...'''\n") # Debug: Print start of chunk text
            self.generate_speech(chunk, str(output_file_path)) # Pass path as string
            # Check if file was actually created (especially if fallback happened)
            final_output_path = output_file_path
            if not output_file_path.exists() and output_file_path.with_suffix('.wav').exists():
                 final_output_path = output_file_path.with_suffix('.wav') # Update if fallback created WAV

            if final_output_path.exists():
                output_files.append(str(final_output_path))
            else:
                 print(f"WARNING (TTS): Output file was not created for chunk {i+1}: {output_file_path.name}")

            # Small delay
            time.sleep(0.5)

        print(f"\nFinished TTS processing for {md_file_path_obj.name}. Generated {len(output_files)} audio file(s).")
        return output_files


# --- OptimizedTTSGenerator Class (Remains Unchanged) ---
# This class orchestrates calling TTSGenerator for multiple files/dirs
class OptimizedTTSGenerator:
    """ Processes markdown files from nested directories """
    def __init__(self, input_dir='processed_docs_output2', output_dir='tts_output', engine_type="gtts"):
        self.input_dir = Path(input_dir) # Use Path
        self.output_dir = Path(output_dir) # Use Path
        # Initialize TTSGenerator with the chosen engine
        self.tts_generator = TTSGenerator(engine_type=engine_type)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Audio files will be saved under: {self.output_dir.resolve()}")

    def process_all_markdown_files(self):
        """ Process all _book_like.md files """
        if not self.input_dir.is_dir():
            print(f"Error: Input directory not found: {self.input_dir}")
            return []

        # Find all subdirectories directly under input_dir
        try:
            book_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"Error listing directories in {self.input_dir}: {e}")
            return []

        if not book_dirs:
            print(f"No subdirectories (book directories) found in {self.input_dir}")
            return []

        print(f"Found {len(book_dirs)} potential book directories to process.")
        all_output_files = []

        for book_dir_path in book_dirs:
            book_name = book_dir_path.name # Get the directory name
            print(f"\n--- Processing book directory: {book_name} ---")
            # Define the specific output directory for this book's audio
            book_output_dir_path = self.output_dir / book_name
            book_output_dir_path.mkdir(parents=True, exist_ok=True)

            # Find the specific markdown file within this book directory
            try:
                md_files = list(book_dir_path.glob('*_book_like.md'))
            except Exception as e:
                print(f"Error scanning for markdown files in {book_dir_path}: {e}")
                continue # Skip to next book directory

            if not md_files:
                print(f"No '*_book_like.md' files found in {book_dir_path}")
                continue

            if len(md_files) > 1:
                 print(f"WARNING: Found multiple '*_book_like.md' files in {book_dir_path}. Processing only the first one: {md_files[0].name}")

            md_file_path = md_files[0] # Process the first one found
            print(f"Processing Markdown file: {md_file_path.name}")

            # Call the TTS processing method for the single markdown file
            audio_files = self.tts_generator.process_markdown_to_speech(
                str(md_file_path), # Pass path as string
                output_dir=str(book_output_dir_path), # Pass path as string
                # chunk_size can be passed from config if needed
            )
            all_output_files.extend(audio_files)

        return all_output_files

    # process_specific_book can remain similar if needed, just ensure paths are correct

# --- main Function (For standalone execution, Remains Mostly Unchanged) ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert markdown to speech")
    # ... (argparse setup remains the same) ...
    args = parser.parse_args()

    # Use OptimizedTTSGenerator for processing directories
    tts_processor = OptimizedTTSGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        engine_type=args.engine
    )

    if args.book:
        # Assuming process_specific_book is implemented similarly to process_all_...
        # output_files = tts_processor.process_specific_book(args.book)
        print("Processing specific book is not fully implemented in this example.")
        output_files = [] # Placeholder
    else:
        output_files = tts_processor.process_all_markdown_files()

    if output_files:
        print(f"\nGenerated {len(output_files)} audio files:")
        for file in output_files: print(f"- {file}")
    else:
        print("No audio files were generated.")

if __name__ == "__main__":
    main()