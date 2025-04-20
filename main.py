# main.py
"""
Main entry point for the modular document processing pipeline.
Orchestrates dependency checks, configuration, initialization,
batch processing, and reporting.
"""

import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional # <--- IMPORT Optional HERE

# Import project modules
import config           # Configuration variables
import dependencies     # Dependency installation/check
import output_saver     # Functions for saving (used indirectly via handler)
# Import the pipeline class - Ensure docling_handler.py is in the same directory or Python path
try:
    from docling_handler import DocumentParserPipeline
except ImportError as e:
    print(f"ERROR: Failed to import DocumentParserPipeline from docling_handler: {e}")
    print("Ensure docling_handler.py is in the correct location and has no errors.")
    sys.exit(1)

# --- Global Check ---
# It's useful to know if optional dependencies are available early
isOpenpyxlAvailable = False

# --- Helper Functions ---

def check_optional_dependencies():
    """Checks for optional dependencies like openpyxl."""
    global isOpenpyxlAvailable
    try:
        import openpyxl
        isOpenpyxlAvailable = True
        print("✅ openpyxl found (optional, for .xlsx output).")
    except ImportError:
        isOpenpyxlAvailable = False
        print("⚠️ Warning: openpyxl not found. .xlsx output for tables will be skipped.")

def setup_directories(input_dir: Path, output_dir: Path) -> bool:
    """Creates input and output directories if they don't exist."""
    print("\n--- Preparing Directories ---")
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Input directory ready: '{input_dir.resolve()}'")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: '{output_dir.resolve()}'")
        return True
    except Exception as e:
        print(f"❌ ERROR: Could not create directories: {e}")
        return False

def find_documents(input_dir: Path, supported_extensions: set) -> List[Path]:
    """Scans the input directory for processable documents."""
    print("\n--- Scanning Input Directory ---")
    documents_to_process = []
    skipped_files = []
    try:
        # Ensure input directory exists before iterating
        if not input_dir.is_dir():
             print(f"❌ ERROR: Input directory '{input_dir}' does not exist or is not a directory.")
             return [] # Return empty list

        all_files = list(input_dir.iterdir())
        documents_to_process = sorted([
            p for p in all_files if p.is_file()
            and p.suffix.lower() in supported_extensions
            and not p.name.startswith('~$') # Skip temp Word/Excel files
        ])
        skipped_files = [p for p in all_files if p.is_file() and p not in documents_to_process]

        print(f"Found {len(documents_to_process)} supported documents ({', '.join(supported_extensions)}).")
        if skipped_files:
             print(f"  (Skipped {len(skipped_files)} non-supported or temporary files)")

        if documents_to_process:
            print("Documents to process:")
            for doc in documents_to_process: print(f"  - {doc.name}")
        else:
             print(f"\n⚠️ No supported documents found in '{input_dir.name}'.")

    except Exception as e:
        print(f"❌ ERROR scanning input directory '{input_dir}': {e}")
        traceback.print_exc() # Print stack trace for debugging scanning issues

    return documents_to_process

def display_configuration():
    """Prints the current runtime configuration."""
    print("\n--- Runtime Configuration ---")
    print(f"  Input Directory:  '{config.INPUT_DIR.resolve()}'")
    # Ensure output dir exists before resolving if it's created later
    try:
        output_dir_resolved = Path(config.CONFIG_OUTPUT_DIR).resolve()
    except FileNotFoundError:
        output_dir_resolved = Path(config.CONFIG_OUTPUT_DIR).absolute() # Show absolute path if not created yet
    print(f"  Output Directory: '{output_dir_resolved}'")
    print(f"  OCR Engine:       {'Disabled' if config.CONFIG_OCR_ENGINE is None else config.CONFIG_OCR_ENGINE}")
    print(f"  Processing Device:{config.CONFIG_DEVICE}")
    print(f"  Image Scale:      {config.CONFIG_IMAGE_SCALE}")
    print(f"  Supported Types:  {', '.join(config.SUPPORTED_EXTENSIONS)}")
    print(f"  XLSX Output:      {'Enabled' if isOpenpyxlAvailable else 'Disabled'}")
    print("-" * 35)

def run_batch_processing(pipeline: DocumentParserPipeline, doc_paths: List[Path]):
    """Processes a list of documents using the pipeline instance."""
    total_docs = len(doc_paths)
    if total_docs == 0:
        print("No documents to process.")
        return

    print(f"\n--- Starting Batch Processing ({total_docs} documents) ---")
    success_count = 0
    failure_count = 0
    batch_start_time = time.time()

    for i, doc_path in enumerate(doc_paths):
        print(f"\n--- Document {i+1}/{total_docs} ---")
        try:
            # process_document now returns True for success/partial, False for failure
            if pipeline.process_document(doc_path):
                success_count += 1
            else:
                failure_count += 1
        except Exception as batch_err:
             failure_count += 1
             print(f"   [CRITICAL BATCH ERROR] Unhandled exception during process_document call for {doc_path.name}: {batch_err}")
             traceback.print_exc()
             # Attempt to save metadata for this critical failure if possible
             # Ensure pipeline and output_root exist before using them
             if pipeline and hasattr(pipeline, 'output_root'):
                 doc_output_dir = pipeline.output_root / doc_path.stem
                 try:
                     doc_output_dir.mkdir(parents=True, exist_ok=True)
                     # Ensure output_saver and its functions are available
                     if 'output_saver' in sys.modules and hasattr(output_saver, 'save_processing_metadata'):
                         pipeline_config = pipeline.get_pipeline_config_dict() # Get config
                         error_info = [{"code": "BATCH_PROCESSING_EXCEPTION", "message": f"{type(batch_err).__name__}: {str(batch_err)}"}]
                         output_saver.save_processing_metadata( # Use saver directly for metadata
                             output_dir=doc_output_dir, base_filename=doc_path.stem, input_suffix=doc_path.suffix,
                             status="BATCH_PROCESSING_EXCEPTION", start_time=time.time(), end_time=time.time(),
                             counts={"text_formats":{}, "images":0, "tables":0, "integrated_markdown":0}, # Default counts
                             pipeline_config=pipeline_config, errors=error_info
                         )
                     else:
                          print("ERROR: output_saver module or function not loaded, cannot save batch error metadata.")
                 except Exception as meta_err:
                     print(f"ERROR: Failed to save error metadata after critical batch error for {doc_path.name}: {meta_err}")
             else:
                 print("ERROR: Pipeline object not fully initialized, cannot determine output path for error metadata.")


    batch_end_time = time.time()
    total_duration = batch_end_time - batch_start_time
    avg_time = (total_duration / total_docs) if total_docs > 0 else 0

    print("\n--- Batch Processing Complete ---")
    print(f"  Total documents attempted: {total_docs}")
    print(f"  Successful (incl. partial w/ output): {success_count}")
    print(f"  Failures / Errors preventing output: {failure_count}")
    print(f"  Total batch time: {total_duration:.2f} seconds")
    avg_time_str = f"({avg_time:.2f}s avg/doc)" if total_docs > 0 else ""
    # Ensure pipeline exists before resolving output path
    output_path_final = Path(config.CONFIG_OUTPUT_DIR).resolve() if not pipeline else pipeline.output_root.resolve()
    print(f"  Output generated in: '{output_path_final}' {avg_time_str}")


# --- Main Execution ---
if __name__ == "__main__":
    start_overall_time = time.time()
    print("\n" + "="*50)
    print("   Document Parsing Pipeline Script (No OCR - Modular)")
    print("="*50 + "\n")

    # 1. Dependency Check
    required_pkgs = ['Pillow', 'pandas', 'openpyxl', 'docling']
    print("Step 1: Checking Dependencies...")
    # Ensure dependencies module and function are available
    if 'dependencies' in sys.modules and hasattr(dependencies, 'install_and_check'):
        if not dependencies.install_and_check(required_pkgs):
            print("\nDependency check failed. Please resolve issues and try again.")
            sys.exit(1) # Exit if dependencies failed critically
    else:
         print("ERROR: dependencies module not loaded correctly. Cannot check packages.")
         sys.exit(1)

    # 2. Check Optional Dependencies
    print("\nStep 2: Checking Optional Dependencies...")
    check_optional_dependencies()

    # 3. Setup Directories
    print("\nStep 3: Setting Up Directories...")
    if not setup_directories(config.INPUT_DIR, Path(config.CONFIG_OUTPUT_DIR)):
        sys.exit(1)

    # 4. Display Configuration
    print("\nStep 4: Displaying Configuration...")
    display_configuration()

    # 5. Find Documents
    print("\nStep 5: Finding Documents...")
    documents_to_process = find_documents(config.INPUT_DIR, config.SUPPORTED_EXTENSIONS)
    if not documents_to_process:
         print("\nNo documents to process. Exiting.")
         sys.exit(0)

    # 6. Initialize Pipeline
    pipeline_instance: Optional[DocumentParserPipeline] = None
    print("\nStep 6: Initializing Document Parser Pipeline...")
    try:
        # --- Map file extensions to Docling InputFormat Enums ---
        # This is crucial because DocumentConverter expects InputFormat enums, not just strings
        try:
            from docling.datamodel.base_models import InputFormat
            format_map = {
                ".pdf": InputFormat.PDF,
                ".docx": InputFormat.DOCX,
                # Add mappings for other supported extensions if needed
                # e.g., ".html": InputFormat.HTML, ".md": InputFormat.MD
            }
            allowed_formats_enum = []
            unsupported_requested = []
            for ext in config.SUPPORTED_EXTENSIONS:
                 if ext in format_map:
                     allowed_formats_enum.append(format_map[ext])
                 else:
                      unsupported_requested.append(ext)
            if unsupported_requested:
                 print(f"WARNING: Extensions {unsupported_requested} in config are not currently mapped to Docling InputFormat enums in main.py and will be ignored.")
            if not allowed_formats_enum:
                 print("ERROR: No supported extensions in config could be mapped to Docling InputFormat. Cannot initialize pipeline.")
                 sys.exit(1)
            print(f"Mapped extensions to InputFormats: {[f.name for f in allowed_formats_enum]}")

        except ImportError:
             print("ERROR: Failed to import InputFormat from docling. Cannot map extensions.")
             print("Ensure 'docling' is installed correctly.")
             sys.exit(1)
        except Exception as e_map:
            print(f"ERROR during extension mapping: {e_map}")
            sys.exit(1)
        # --- End Mapping ---


        pipeline_instance = DocumentParserPipeline(
            output_root=config.CONFIG_OUTPUT_DIR,
            device=config.CONFIG_DEVICE,
            image_scale=config.CONFIG_IMAGE_SCALE,
            allowed_formats=allowed_formats_enum, # Pass the list of InputFormat enums
            isOpenpyxlAvailable=isOpenpyxlAvailable
        )

    except Exception as setup_error:
        print(f"\n❌ CRITICAL ERROR: Pipeline initialization failed: {setup_error}")
        traceback.print_exc()
        sys.exit(1)

    # 7. Run Batch Processing
    print("\nStep 7: Starting Batch Processing...")
    if pipeline_instance:
         try:
             run_batch_processing(pipeline_instance, documents_to_process)
         except Exception as batch_run_error:
              print(f"\n❌ CRITICAL ERROR during batch processing run: {batch_run_error}")
              traceback.print_exc()
    else:
         print("ERROR: Pipeline instance not created. Cannot run batch processing.")


    # 8. Finish
    end_overall_time = time.time()
    print("\n" + "="*50)
    print(f"   Script Finished (Total Time: {(end_overall_time - start_overall_time):.2f}s)")
    print("="*50 + "\n")
