import sys
import time
import traceback
import logging
from pathlib import Path

import argparse
import subprocess

# Import configuration first
import pipeline_config as cfg

# Import pipeline step modules and utility functions
import dependencies
import pipeline_utils
try:
    from docling_handler import DocumentParserPipeline
    import markdown_reformatter
    import image_descriptor
    import transcript
    # import convert_to_ir_chunked # REMOVED
    import tts
    # voice_engine_cli is run as subprocess, no direct import needed unless refactored differently
except ImportError as e:
    print(f"FATAL ERROR: Failed to import a required pipeline module: {e}")
    print("Please ensure all script files (docling_handler.py, etc.) are present and dependencies are installed.")
    # Add specific check if a needed module failed
    if 'convert_to_ir_chunked' not in str(e): # Only exit if a *required* module is missing
        sys.exit(1)
    else:
        # If it was just the removed module, we can continue, but log it
        print("INFO: Could not import 'convert_to_ir_chunked' - this module has been removed from the pipeline.")


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)-7s] %(module)-20s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# --- Define Output Structure ---
def get_doc_paths(base_output_dir: Path, doc_stem: str) -> dict[str, Path]:
    """Generates standard paths for a document's outputs."""
    doc_output_root = base_output_dir / doc_stem
    return {
        "root": doc_output_root,
        "docling_output": doc_output_root / "docling_raw", # Specific subdir for docling's direct outputs
        "book_like_md": doc_output_root / f"{doc_stem}{cfg.DOCLING_BOOK_LIKE_MD_SUFFIX}",
        "formatted_md": doc_output_root / f"{doc_stem}{cfg.FORMATTED_MD_SUFFIX}",
        "image_desc_json": doc_output_root / f"{doc_stem}{cfg.IMAGE_DESC_JSON_SUFFIX}",
        "transcript_json": doc_output_root / f"{doc_stem}{cfg.TRANSCRIPT_JSON_SUFFIX}",
        # "ir_json": doc_output_root / f"{doc_stem}{cfg.IR_JSON_SUFFIX}", # REMOVED
        "tts_audio_dir": doc_output_root / cfg.TTS_OUTPUT_SUBDIR,
        "processing_metadata": doc_output_root / f"{doc_stem}_processing_metadata.json", # Keep metadata at doc root
    }

# --- Main Pipeline Function ---
def run_pipeline():
    log.info("=" * 60)
    log.info(" Starting Exam Processing Pipeline ".center(60, "="))
    log.info("=" * 60)
    overall_start_time = time.time()

    # --- Argument Parsing (Optional Overrides) ---
    parser = argparse.ArgumentParser(description="Run the full Exam Processing Pipeline.")
    parser.add_argument("--input-dir", help=f"Override INPUT_DIR (Default: {cfg.INPUT_DIR})")
    parser.add_argument("--output-dir", help=f"Override BASE_OUTPUT_DIR (Default: {cfg.BASE_OUTPUT_DIR})")
    args = parser.parse_args()

    # Update config if overrides provided
    if args.input_dir: cfg.INPUT_DIR = args.input_dir
    if args.output_dir: cfg.BASE_OUTPUT_DIR = args.output_dir

    # 1. Dependency Check
    log.info("Step 1: Checking Dependencies...")
    # Removed 'jsonschema' as it was likely primarily for IR validation
    required_pkgs = ['Pillow', 'pandas', 'openpyxl', 'docling', 'requests', 'gTTS', 'pygame', 'SpeechRecognition']
    if not dependencies.install_and_check(required_pkgs):
        log.error("Core dependency check failed. Exiting.")
        sys.exit(1)
    # Check optional (won't exit if missing, just warns)
    opt_deps_status = pipeline_utils.check_optional_dependencies(['openpyxl'])
    isOpenpyxlAvailable = opt_deps_status.get('openpyxl', False)
    log.info("Dependency checks complete.")

    # 2. Setup Directories & Config Display
    log.info("Step 2: Setting Up Directories & Configuration...")
    if not pipeline_utils.setup_directories(cfg.get_config_dict()): # Pass config dict
        sys.exit(1)
    pipeline_utils.display_configuration(cfg.get_config_dict()) # Display effective config

    # 3. Find Documents
    log.info("Step 3: Finding documents...")
    input_dir = Path(cfg.INPUT_DIR)
    documents_to_process = pipeline_utils.find_documents(input_dir, cfg.SUPPORTED_EXTENSIONS)
    if not documents_to_process:
        log.warning(f"No supported documents found in {input_dir}. Exiting.")
        sys.exit(0)
    log.info(f"Found {len(documents_to_process)} documents to process.")

    # 4. Initialize Docling Pipeline (once)
    log.info("Step 4: Initializing Docling Parser...")
    docling_pipeline = None
    try:
        from docling.datamodel.base_models import InputFormat
        format_map = { ".pdf": InputFormat.PDF, ".docx": InputFormat.DOCX }
        allowed_formats_enum = [format_map[ext] for ext in cfg.SUPPORTED_EXTENSIONS if ext in format_map]
        if not allowed_formats_enum:
            raise ValueError("No supported extensions map to Docling InputFormats.")

        docling_pipeline = DocumentParserPipeline(
            base_output_dir=Path(cfg.BASE_OUTPUT_DIR),
            device=cfg.DOCLING_DEVICE,
            image_scale=cfg.DOCLING_IMAGE_SCALE,
            allowed_formats=allowed_formats_enum,
            isOpenpyxlAvailable=isOpenpyxlAvailable
        )
        log.info("Docling Parser initialized successfully.")
    except Exception:
        log.exception("CRITICAL: Failed to initialize Docling pipeline.")
        sys.exit(1)

    # 5. Process Each Document
    log.info("Step 5: Processing documents...")
    successful_docs_paths = {} # Store paths for last successful doc
    failed_docs = []

    for doc_path in documents_to_process:
        doc_start_time = time.time()
        doc_stem = doc_path.stem
        log.info(f"--- Processing document: {doc_path.name} ---")
        paths = get_doc_paths(Path(cfg.BASE_OUTPUT_DIR), doc_stem)
        current_step = "Initialization"
        doc_success = False
        last_generated_md_path = None # Track the input for downstream steps
        last_image_desc_path = None

        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["docling_output"].mkdir(parents=True, exist_ok=True)

        try:
            # === Step 5.1: Docling Conversion ===
            current_step = "Docling Conversion"
            log.info(f"  Running Docling conversion -> {paths['docling_output'].name} & {paths['book_like_md'].name}...")
            docling_result = docling_pipeline.process_document(doc_path, paths['docling_output'])

            if not docling_result.get("success"):
                log.error(f"  Docling processing failed. Status: {docling_result.get('status', 'Unknown')}")
                docling_errors = docling_result.get('errors', [])
                if docling_errors:
                    log.error(f"  Docling Errors/Warnings:")
                    for err in docling_errors:
                        code = getattr(err, 'error_code', 'N/A')
                        msg = getattr(err, 'error_message', str(err))
                        log.error(f"    - Code: {code}, Message: {msg}")
                raise RuntimeError("Docling processing failed")

            book_like_md_path = paths["book_like_md"]
            if not book_like_md_path.exists():
                 log.error(f"  Expected Book-like markdown not found after Docling: {book_like_md_path}")
                 if "FAILURE_MD_GENERATION" in docling_result.get('status', ''):
                      log.error("  Docling result indicates markdown generation failure.")
                 raise RuntimeError("Book-like markdown generation failed or was not created")
            log.info(f"  Docling step successful. Initial MD: {book_like_md_path.name}")
            last_generated_md_path = book_like_md_path


            # === Step 5.2: Markdown Reformatting (Optional) ===
            current_step = "Markdown Reformatting"
            if cfg.SKIP_MARKDOWN_REFORMATTING:
                log.info(f"  Skipping {current_step} as per configuration.")
                last_generated_md_path = book_like_md_path
            else:
                log.info(f"  Running {current_step} using LLM -> {paths['formatted_md'].name}...")
                reformat_success = markdown_reformatter.format_markdown_properly(
                    input_path=book_like_md_path,
                    output_path=paths['formatted_md'],
                    config=cfg.get_config_dict()
                )
                if not reformat_success:
                    raise RuntimeError("Markdown reformatting failed")
                log.info(f"  Reformatting successful.")
                last_generated_md_path = paths['formatted_md']

            # === Step 5.3: Image Description (Optional) ===
            current_step = "Image Description"
            if cfg.SKIP_IMAGE_DESCRIPTION:
                log.info(f"  Skipping {current_step} as per configuration.")
                last_image_desc_path = None
            else:
                if not last_generated_md_path:
                     raise RuntimeError("Cannot run Image Description without a valid Markdown input path.")
                log.info(f"  Running {current_step} using LLaVA -> {paths['image_desc_json'].name}...")
                desc_success = image_descriptor.generate_descriptions_for_doc(
                    markdown_file_path=last_generated_md_path,
                    output_json_path=paths['image_desc_json'],
                    config=cfg.get_config_dict()
                )
                if not desc_success:
                    raise RuntimeError("Image description generation failed")
                log.info(f"  Image descriptions generated.")
                last_image_desc_path = paths['image_desc_json']

            # === Step 5.4: Transcript Generation (Optional) ===
            current_step = "Transcript Generation"
            if cfg.SKIP_TRANSCRIPT_GENERATION:
                log.info(f"  Skipping {current_step} as per configuration.")
            else:
                if not last_generated_md_path:
                     raise RuntimeError("Cannot run Transcript Generation without a valid Markdown input path.")
                log.info(f"  Running {current_step} using LLM -> {paths['transcript_json'].name}...")
                transcript_success = transcript.generate_transcripts_via_llm(
                    input_md_file=last_generated_md_path,
                    image_descriptions_file=last_image_desc_path, # Can be None if skipped
                    output_json_file=paths['transcript_json'],
                    config=cfg.get_config_dict()
                )
                if not transcript_success:
                    raise RuntimeError("Transcript generation failed")
                log.info(f"  Transcript generation successful.")


            # === Step 5.5: Intermediate Representation (IR) Generation (REMOVED) ===
            # current_step = "IR Generation"
            # if cfg.SKIP_IR_GENERATION:
            #      log.info(f"  Skipping {current_step} as per configuration.")
            # else:
            #     # ... (removed code block) ...
            #     log.info(f"  IR generation successful.")
            log.info("  Skipping Intermediate Representation (IR) generation step (removed).")


            # === Step 5.6 -> 5.5: Text-to-Speech (TTS) (Optional) ===
            current_step = "TTS Generation"
            if not cfg.ENABLE_TTS: # Check flag from config
                log.info(f"  Skipping {current_step} as per configuration.")
            else:
                transcript_json_path = paths['transcript_json']
                if not transcript_json_path.exists():
                    log.warning(f"  Skipping {current_step}: Transcript JSON not found ({transcript_json_path.name}). Was transcript generation skipped or failed?")
                else:
                    log.info(f"  Running {current_step} using gTTS -> {paths['tts_audio_dir'].name}...")
                    tts_success = tts.generate_audio_files(
                        transcript_json_path=transcript_json_path,
                        output_audio_dir=paths['tts_audio_dir'],
                        config=cfg.get_config_dict()
                    )
                    if not tts_success:
                        # Decide if this is critical? Maybe not.
                        log.warning("  TTS generation process reported potential issues (e.g., gTTS errors, empty file) but continuing pipeline.")
                    else:
                        log.info(f"  TTS generation successful.")

            # If all steps (or those not skipped) succeeded
            doc_success = True
            successful_docs_paths[doc_stem] = paths # Store paths of the last successful doc
            log.info(f"--- Successfully processed {doc_path.name} in {(time.time() - doc_start_time):.2f}s ---")

        except Exception as e:
            doc_success = False
            failed_docs.append(doc_path.name)
            log.error(f"--- FAILED processing {doc_path.name} at step '{current_step}' ---")
            log.exception(f"  Error details") # Logs full traceback

    # 6. Final Summary
    log.info("=" * 60)
    log.info(" Pipeline Finished ".center(60, "="))
    log.info(f"Total execution time: {(time.time() - overall_start_time):.2f}s")
    log.info(f"Successfully processed: {len(successful_docs_paths)} ({', '.join(successful_docs_paths.keys())})")
    log.info(f"Failed documents: {len(failed_docs)} ({', '.join(failed_docs)})")
    log.info(f"Outputs located in base directory: {Path(cfg.BASE_OUTPUT_DIR).resolve()}")
    log.info("=" * 60)

    # 7. Optional: Launch Voice Engine for the *last successfully processed* document
    # This part remains the same as it already uses transcript_json and tts_audio_dir
    if cfg.LAUNCH_VOICE_ENGINE_AFTER and successful_docs_paths:
        last_success_stem = list(successful_docs_paths.keys())[-1]
        last_paths = successful_docs_paths[last_success_stem]
        transcript_path = last_paths['transcript_json']
        audio_path = last_paths['tts_audio_dir'] # Use the TTS audio dir

        if transcript_path.exists() and audio_path.exists() and audio_path.is_dir():
             log.info(f"Launching Voice Engine for: {last_success_stem}...")
             voice_engine_script = Path(__file__).parent / "voice_engine_cli.py" # Assume it's in the same dir
             if not voice_engine_script.exists():
                  log.error(f"Cannot launch voice engine: Script not found at {voice_engine_script}")
             else:
                 cmd = [
                     sys.executable, # Use the same python interpreter
                     str(voice_engine_script),
                     "--transcript-json", str(transcript_path),
                     "--audio-dir", str(audio_path)
                 ]
                 log.info(f"Executing: {' '.join(cmd)}")
                 try:
                      subprocess.run(cmd, check=True)
                 except subprocess.CalledProcessError as e:
                      log.error(f"Voice engine process failed with exit code {e.returncode}")
                 except FileNotFoundError:
                      log.error(f"Could not find voice_engine_cli.py or python executable.")
                 except Exception as ve_err:
                      log.error(f"Failed to launch voice engine subprocess: {ve_err}")
        else:
             log.warning(f"Cannot launch voice engine: Required files/dirs not found for {last_success_stem}")
             log.warning(f"  Checked Transcript: {transcript_path} (Exists: {transcript_path.exists()})")
             log.warning(f"  Checked Audio Dir: {audio_path} (Exists: {audio_path.exists()}, IsDir: {audio_path.is_dir()})")


if __name__ == "__main__":
    run_pipeline()