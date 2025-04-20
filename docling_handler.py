# docling_handler.py
"""
Contains the DocumentParserPipeline class for handling docling conversion
and orchestrating the saving of outputs for a single document.
"""
import time
import os
import logging # <<< Added
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import project modules & config
# <<< NOTE: pipeline_config is not directly used in this file anymore, but Timeout was used via cfg >>>
# <<< We'll remove the direct import if not needed elsewhere, but keep cfg usage for now >>>
import pipeline_config as cfg # Still needed for timeout value *retrieval* even if not passed to convert
import output_saver
import markdown_generator

# Setup logger for this module
log = logging.getLogger(__name__) # <<< Added

# --- Docling Imports ---
try:
    from docling.datamodel.base_models import (
        InputFormat, ConversionStatus, ErrorItem # Make sure ErrorItem is imported
    )
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    )
    from docling.document_converter import (
        DocumentConverter, PdfFormatOption, ConversionResult, FormatOption
    )
    log.info("Successfully imported required Docling components.") # <<< Changed
except ImportError as e:
    # Log critical error and re-raise to stop initialization if Docling is missing
    log.exception(f"CRITICAL ERROR: Failed importing essential Docling components: {e}") # <<< Changed
    # Define dummy ErrorItem if import fails to prevent NameError later, though functionality will be limited
    class ErrorItem:
         def __init__(self, component_type, module_name, error_code, error_message, **kwargs):
             self.component_type = component_type
             self.module_name = module_name
             self.error_code = error_code
             self.error_message = error_message
    log.warning("Using dummy ErrorItem due to import failure.")
    # raise ImportError("Failed importing essential Docling components. Ensure 'docling' is installed.") from e # Keep this commented if you want to try running with dummy
except Exception as e:
    log.exception(f"CRITICAL ERROR: Unexpected error during Docling import: {e}") # <<< Changed
    # Define dummy ErrorItem here too
    class ErrorItem:
         def __init__(self, component_type, module_name, error_code, error_message, **kwargs):
             self.component_type = component_type
             self.module_name = module_name
             self.error_code = error_code
             self.error_message = error_message
    log.warning("Using dummy ErrorItem due to import failure.")
    # raise RuntimeError(f"Unexpected error during Docling import: {e}") from e # Keep this commented if you want to try running with dummy
# --- End Docling Imports ---


class DocumentParserPipeline:
    """
    Handles document conversion using Docling and saves structured outputs.
    OCR is disabled in this configuration.
    """
    def __init__(
        self,
        base_output_dir: Path, # Takes the base output directory
        device: str,
        image_scale: float,
        allowed_formats: List[InputFormat],
        isOpenpyxlAvailable: bool,
        # Removed output_root, replaced with base_output_dir
    ):
        """
        Initializes the pipeline components (converter, accelerator).
        The actual output directory is determined per document in process_document.

        Args:
            base_output_dir: The root directory for all pipeline outputs (used for logging context).
            device: Processing device ('auto', 'cpu', 'cuda', 'mps').
            image_scale: Scaling factor for extracted images.
            allowed_formats: List of docling InputFormat enums to allow.
            isOpenpyxlAvailable: Boolean indicating if openpyxl is installed.
        """
        self.base_output_dir = base_output_dir # Store for reference
        self.device = device.lower()
        self.image_scale = image_scale
        self.allowed_formats = allowed_formats
        self.isOpenpyxlAvailable = isOpenpyxlAvailable

        log.info("Initializing Document Parser Pipeline...")
        log.info(f"  Base Output Dir: {self.base_output_dir.resolve()}")
        log.info(f"  Device: {self.device}, Image Scale: {self.image_scale}")
        log.info(f"  Allowed Formats: {[f.name for f in self.allowed_formats]}")
        log.info(f"  XLSX Output Enabled: {self.isOpenpyxlAvailable}")

        self.accelerator_options = self._configure_accelerator()
        self.converter: Optional[DocumentConverter] = None
        self.converter = self._configure_converter() # Can raise exceptions

        if self.converter:
             log.info("Docling Converter Initialized Successfully.")
        else:
             # Error should have been raised in _configure_converter
             log.critical("Pipeline initialization failed: Converter is None after configuration.")
             raise RuntimeError("Pipeline initialization failed: Converter setup returned None.")


    def _configure_accelerator(self) -> AcceleratorOptions:
        """Configures accelerator options based on the device setting."""
        log.info("Configuring accelerator...")
        selected_device_enum = AcceleratorDevice.AUTO # Default
        try:
            device_map = {
                'cpu': AcceleratorDevice.CPU, 'cuda': AcceleratorDevice.CUDA,
                'gpu': AcceleratorDevice.CUDA, 'mps': AcceleratorDevice.MPS,
                'auto': AcceleratorDevice.AUTO,
            }
            if self.device == 'mps' and not hasattr(AcceleratorDevice, 'MPS'):
                log.warning("Device 'mps' requested, but not available in this Docling version. Falling back to 'auto'.")
                selected_device_enum = AcceleratorDevice.AUTO
            elif self.device in device_map:
                selected_device_enum = device_map[self.device]
            else:
                log.warning(f"Invalid device string '{self.device}'. Falling back to 'auto'. Valid: {list(device_map.keys())}")
                selected_device_enum = AcceleratorDevice.AUTO
        except Exception as e:
             log.error(f"Error configuring accelerator device '{self.device}': {e}. Falling back to 'auto'.")
             selected_device_enum = AcceleratorDevice.AUTO

        log.info(f"Accelerator configured for device: {selected_device_enum.name}")
        num_threads = os.cpu_count()
        if num_threads is None or num_threads < 1: num_threads = 4
        log.info(f"Using {num_threads} threads for accelerator options.")
        return AcceleratorOptions(num_threads=num_threads, device=selected_device_enum)

    def _configure_converter(self) -> DocumentConverter:
        """Configures and returns the Docling DocumentConverter instance."""
        log.info("Configuring Docling Converter (OCR Disabled)...")
        try:
            common_pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                table_structure_options={"do_cell_matching": True},
                images_scale=self.image_scale,
                generate_picture_images=True,
                accelerator_options=self.accelerator_options
            )
            format_options: Dict[InputFormat, FormatOption] = {}
            for fmt in self.allowed_formats:
                if fmt in [InputFormat.PDF]:
                    format_options[fmt] = PdfFormatOption(pipeline_options=common_pipeline_options)
                    log.info(f"  Applied PDF/Image options to {fmt.name}")
                elif fmt == InputFormat.DOCX:
                    # Note: Using PdfFormatOption for DOCX might have limitations or unexpected behavior.
                    # Check Docling documentation if specific DocxFormatOption exists or is needed.
                    # Assuming PdfFormatOption works generically here.
                    format_options[fmt] = PdfFormatOption(pipeline_options=common_pipeline_options)
                    log.info(f"  Applied generic options to {fmt.name} (using PdfFormatOption)")
                else:
                     log.warning(f"  No specific FormatOption configured for {fmt.name}. Using Docling defaults if available.")

            if not format_options:
                 log.error("No format options could be configured based on allowed_formats.")
                 raise ValueError("No format options could be configured based on allowed_formats.")

            converter = DocumentConverter(
                allowed_formats=self.allowed_formats,
                format_options=format_options,
            )
            log.info(f"Converter ready for formats: {[f.name for f in converter.allowed_formats]}")
            log.info(f"Pipeline Options Applied (where applicable): Scale={self.image_scale}, Tables=Yes, Images=Yes, OCR=No")
            return converter

        except Exception as e:
            log.exception(f"CRITICAL ERROR: Failed to initialize DocumentConverter: {e}")
            raise # Re-raise the exception to halt pipeline startup


    def get_pipeline_config_dict(self) -> Dict:
        """Returns a dictionary representation of the pipeline configuration for metadata."""
        accel_name = 'N/A'
        try: # Add extra safety
            if self.accelerator_options and hasattr(self.accelerator_options, 'device') and hasattr(self.accelerator_options.device, 'name'):
                 accel_name = self.accelerator_options.device.name
        except Exception:
            log.warning("Could not determine accelerator device name for metadata.", exc_info=True)

        return {
            "ocr_engine": "Disabled",
            "ocr_languages": [],
            "device_setting": self.device,
            "accelerator_used": accel_name,
            "image_scale": self.image_scale,
            "allowed_formats": [f.name for f in self.allowed_formats],
        }

    def process_document(self, doc_path: Path, doc_output_dir: Path) -> Dict:
        """
        Processes a single document: converts, saves standard outputs,
        generates book-like markdown, and saves metadata.

        Args:
            doc_path: Path to the input document.
            doc_output_dir: The specific output directory for this document's results
                            (e.g., ./pipeline_output/doc_stem/docling_raw).

        Returns:
            A dictionary containing:
                - success (bool): True if processing was successful enough to continue.
                - status (str): Final status string (e.g., SUCCESS, FAILURE_...).
                - book_like_md_path (Path | None): Path to the generated book-like markdown, or None on failure.
                - errors (List): List of any errors encountered.
        """
        if not self.converter:
             log.error(f"Converter not initialized. Cannot process {doc_path.name}")
             return {"success": False, "status": "FAILURE_NO_CONVERTER", "book_like_md_path": None, "errors": ["Converter not initialized."]}

        start_time = time.time()
        base_filename = doc_path.stem
        # Ensure the specific output directory for this document exists
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Define expected output path for book-like MD (relative to the *parent* of doc_output_dir) ---
        # The Markdown file goes into the document's root folder, not the docling_raw subdir
        book_like_md_path = doc_output_dir.parent / f"{base_filename}{cfg.DOCLING_BOOK_LIKE_MD_SUFFIX}"

        status_str = "INITIATED"
        final_counts = {
            "text_formats": {"docling_markdown": 0, "json": 0, "plain_text": 0},
            "images": 0,
            "tables": 0,
            "book_like_markdown": 0
        }
        conv_errors: List[Any] = [] # Initialize as list
        success_flag = False
        document_obj: Optional[Any] = None
        saved_image_paths: Dict[int, str] = {}
        saved_table_info: Dict[int, str] = {}
        integrated_md_success = False # Track this specific step

        # Use relative path for logging clarity
        log_output_rel_path = doc_output_dir.relative_to(self.base_output_dir) if doc_output_dir.is_relative_to(self.base_output_dir) else doc_output_dir
        log.info(f"Processing: {doc_path.name} -> {log_output_rel_path}")

        if not doc_path.exists():
            log.error(f"Input file not found: {doc_path}. Skipping.")
            status_str = "FAILURE_FILE_NOT_FOUND"
            # <<< FIX 2a: Add required fields to ErrorItem creation >>>
            conv_errors = [ErrorItem(component_type="InputValidation", module_name="docling_handler", error_code=status_str, error_message="Input file path does not exist.") if 'ErrorItem' in globals() and ErrorItem is not None else {"code": status_str, "message": "Input file path does not exist."}]
            success_flag = False
        else:
            try:
                log.info(f"  Converting document...")
                # <<< FIX 1: Remove the unsupported 'timeout' keyword argument >>>
                conv_result: Optional[ConversionResult] = self.converter.convert(doc_path)

                if conv_result is None:
                    raise RuntimeError("DocumentConverter.convert returned None unexpectedly.")

                status_enum = getattr(conv_result, 'status', None)
                if status_enum is None: raise RuntimeError("ConversionResult lacks 'status'.")
                status_str = status_enum.name
                # Ensure errors attribute is a list, default to empty list if missing or None
                conv_errors = getattr(conv_result, 'errors', []) or []

                log.info(f"  Conversion Status: {status_str}")
                if conv_errors:
                     log.warning(f"  Conversion generated {len(conv_errors)} errors/warnings.")
                     for err_item in conv_errors: # Optional detailed logging
                         # Safely access attributes, provide defaults
                         code = getattr(err_item, 'error_code', 'N/A')
                         msg = getattr(err_item, 'error_message', 'N/A')
                         mod = getattr(err_item, 'module_name', 'N/A')
                         comp = getattr(err_item, 'component_type', 'N/A')
                         log.debug(f"    - Code: {code}, Mod: {mod}, Comp: {comp}, Msg: {msg}")

                if status_enum in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
                    document_obj = getattr(conv_result, 'document', None)
                    if document_obj:
                        log.info(f"  Conversion produced document object. Saving outputs...")
                        # --- Call Saver Functions ---
                        try:
                            # Standard outputs (saved within doc_output_dir by saver functions)
                            # Pass the specific docling_raw directory for standard outputs
                            text_counts = output_saver.save_text_outputs(document_obj, doc_output_dir, base_filename)
                            img_count, saved_image_paths = output_saver.save_images(document_obj, doc_output_dir, base_filename)
                            tbl_count, saved_table_info = output_saver.save_tables(document_obj, doc_output_dir, base_filename, self.isOpenpyxlAvailable)

                            final_counts["text_formats"] = text_counts
                            final_counts["images"] = img_count
                            final_counts["tables"] = tbl_count

                            # Book-like markdown generation
                            # Pass the target path explicitly (goes to parent dir of doc_output_dir)
                            integrated_md_success = markdown_generator.create_book_like_markdown(
                                document=document_obj,
                                output_dir=doc_output_dir.parent, # Save MD to the doc's root output dir
                                base_filename=base_filename,
                                saved_image_paths=saved_image_paths,
                                saved_table_info=saved_table_info
                                # Target path is implicitly handled by function now: output_dir / f"{base_filename}_book_like.md"
                            )
                            final_counts["book_like_markdown"] = 1 if integrated_md_success else 0

                            if not integrated_md_success:
                                log.warning("Book-like markdown generation failed.")
                                status_str = "FAILURE_MD_GENERATION" # Set status if MD fails
                                success_flag = False # Mark as failed if MD gen fails

                            else:
                                success_flag = True # Conversion + Output saving + MD worked (partially or fully)

                        except Exception as save_err:
                             log.exception(f"Exception during output saving or MD generation for {doc_path.name}: {save_err}")
                             # <<< FIX 2b: Add required fields to ErrorItem creation >>>
                             err_item = ErrorItem(component_type="OutputSaver", module_name="docling_handler", error_code="OUTPUT_SAVING_ERROR", error_message=str(save_err)) if 'ErrorItem' in globals() and ErrorItem is not None else {"code": "OUTPUT_SAVING_ERROR", "message": str(save_err)}
                             # Ensure conv_errors is a list before appending
                             if not isinstance(conv_errors, list): conv_errors = []
                             conv_errors.append(err_item)
                             success_flag = False # Treat saving error as failure
                             status_str = "FAILURE_DURING_OUTPUT"
                    else:
                        log.error(f"Conversion status {status_str} but no 'document' object returned.")
                        status_str = "FAILURE_NO_DOCUMENT_OBJECT"
                        success_flag = False
                else:
                    log.error(f"Conversion failed with status {status_str}.")
                    success_flag = False # Explicitly set failure flag

            except Exception as e:
                # This catches the initial convert error (like the timeout one)
                # or any other unexpected error during the try block
                status_str = "CRITICAL_PROCESSING_ERROR"
                log.exception(f"CRITICAL ERROR processing {doc_path.name}: {e}")
                # <<< FIX 2c: Add required fields to ErrorItem creation >>>
                # Check if ErrorItem exists and is not None (in case import failed)
                err_item = None
                if 'ErrorItem' in globals() and ErrorItem is not None:
                    try:
                        err_item = ErrorItem(
                            component_type="ProcessingStep",
                            module_name="docling_handler",
                            error_code=status_str,
                            error_message=f"{type(e).__name__}: {str(e)}"
                        )
                    except Exception as item_creation_err:
                        log.error(f"Failed to create ErrorItem instance: {item_creation_err}. Falling back to dict.")
                        err_item = None # Fallback below

                if err_item is None: # Fallback if class missing or instantiation failed
                     err_item = {"code": status_str, "message": f"{type(e).__name__}: {str(e)}"}

                # Ensure conv_errors is a list before appending
                if not isinstance(conv_errors, list): conv_errors = []
                conv_errors.append(err_item)
                success_flag = False

        # --- Finalization ---
        end_time = time.time()
        # Ensure final flag reflects status string semantics
        if "FAILURE" in status_str or "ERROR" in status_str:
             success_flag = False

        pipeline_config_dict = self.get_pipeline_config_dict()

        # Use the specific metadata file path (goes into the document's root output dir)
        metadata_path = doc_output_dir.parent / f"{base_filename}_processing_metadata.json"

        output_saver.save_processing_metadata(
            output_dir=doc_output_dir.parent, # Save metadata to the doc's root output dir
            base_filename=base_filename, # Name for the meta file is based on doc
            input_suffix=doc_path.suffix,
            status=status_str, start_time=start_time, end_time=end_time,
            counts=final_counts,
            pipeline_config=pipeline_config_dict, errors=conv_errors
        )

        status_indicator = "OK    " if success_flag else "FAILED"
        log.info(f"  [{status_indicator}] Finished: {doc_path.name} (Status: {status_str}, {(end_time - start_time):.2f}s)")

        # --- Return Results ---
        # Return the *actual* path where MD was saved (or should have been)
        return {
            "success": success_flag,
            "status": status_str,
            "book_like_md_path": book_like_md_path if success_flag and integrated_md_success else None,
            "errors": conv_errors
        }