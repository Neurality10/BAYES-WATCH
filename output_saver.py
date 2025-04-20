# output_saver.py
"""
Functions for saving standard processed document outputs (text, images, tables, metadata).
Relies on the structure of the 'document' object returned by docling.
Does NOT handle the 'book-like' Markdown generation (see markdown_generator.py).
"""
import json
import re # For image alt text sanitization
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import necessary libraries, handle potential ImportErrors gracefully
try:
    import pandas as pd
except ImportError:
    print("ERROR (output_saver.py): pandas is required but not found. Install it: pip install pandas")
    pd = None # Set to None so checks fail later

try:
    from PIL import Image
except ImportError:
    print("ERROR (output_saver.py): Pillow is required but not found. Install it: pip install Pillow")
    Image = None # Set to None

# Import necessary Docling item types directly here for isinstance checks
# Adapt based on the actual available classes and error messages.
try:
    from docling.datamodel.document import (
        PictureItem, TableItem, TextItem, ListItem # Keep these for type checks
    )
    from docling_core.types.doc import DocItemLabel
    from docling.datamodel.base_models import ErrorItem
    print("INFO (output_saver): Successfully imported required Docling types.")
except ImportError as e:
    print(f"ERROR (output_saver.py): Failed to import required types from docling/docling_core: {e}")
    print("Ensure 'docling' and 'docling_core' are installed correctly.")
    # Define dummy classes so isinstance checks don't raise NameError, but they won't match
    class PictureItem: pass
    class TableItem: pass
    class TextItem: pass # Still needed for save_text_outputs potentially? No, just uses methods.
    class ListItem: pass # Not directly used here anymore
    class ErrorItem: pass # Used in metadata
    class DocItemLabel: # Dummy enum matching expected values
         PAGE_HEADER = "page_header"
         PAGE_FOOTER = "page_footer"
         SECTION_HEADER = "section_header"
         PARAGRAPH = "paragraph"
         TEXT = "text"
         PICTURE = "picture"
         TABLE = "table"
         LIST_ITEM = "list_item"
    print("WARNING (output_saver): Using dummy types due to import failure. Label matching might be affected.")


# --- Text Saving (Standard Formats) ---
def save_text_outputs(document: Any, output_dir: Path, base_filename: str) -> Dict[str, int]:
    """
    Saves standard Docling markdown (if available), JSON, and plain text.
    Note: This saves Docling's default Markdown, not the enhanced 'book-like' version.
    """
    if not document:
        print(f"ERROR (save_text): Document object is None for {base_filename}.")
        return {"docling_markdown": 0, "json": 0, "plain_text": 0} # Key changed for clarity

    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    # Key name updated to avoid confusion with the custom markdown
    counts = {"docling_markdown": 0, "json": 0, "plain_text": 0}
    save_md_method = 'save_as_markdown'
    save_json_method = 'save_as_json'
    plain_text_attr = 'plain_text'

    # Save Standard Docling Markdown (os1_structured.md)
    try:
        md_path = output_dir / "os1_structured.md" # Standard name
        if hasattr(document, save_md_method):
            getattr(document, save_md_method)(md_path)
            counts["docling_markdown"] = 1 # Use updated key
            print(f"  Saved standard Docling markdown: {md_path.name}")
        else:
            print(f"ERROR (save_text): Document object lacks '{save_md_method}' method.")
    except Exception as e:
        print(f"ERROR (save_text): Failed saving standard Docling Markdown for {base_filename}: {e}")

    # Save JSON
    try:
        json_path = text_dir / f"{base_filename}_structured.json"
        if hasattr(document, save_json_method):
            getattr(document, save_json_method)(json_path)
            counts["json"] = 1
            print(f"  Saved structured JSON: {json_path.relative_to(output_dir)}")
        else:
            print(f"ERROR (save_text): Document object lacks '{save_json_method}' method.")
    except Exception as e:
        print(f"ERROR (save_text): Failed saving JSON for {base_filename}: {e}")

    # Save Plain Text
    try:
        txt_path = text_dir / f"{base_filename}_plain.txt"
        saved_plain = False
        if hasattr(document, save_md_method):
            try:
                getattr(document, save_md_method)(txt_path, strict_text=True)
                counts["plain_text"] = 1
                saved_plain = True
                print(f"  Saved plain text (using strict_text): {txt_path.relative_to(output_dir)}")
            except TypeError as te:
                 if 'unexpected keyword argument' in str(te):
                      print(f"INFO (save_text): '{save_md_method}' does not support 'strict_text'. Trying fallback.")
                 else: raise
            except Exception as e_strict:
                 print(f"WARNING (save_text): '{save_md_method}(strict_text=True)' failed: {e_strict}. Trying fallback.")

        if not saved_plain and hasattr(document, plain_text_attr) and isinstance(getattr(document, plain_text_attr, None), str):
             plain_content = getattr(document, plain_text_attr)
             if plain_content:
                 with open(txt_path, 'w', encoding='utf-8') as f: f.write(plain_content)
                 counts["plain_text"] = 1
                 saved_plain = True
                 print(f"  Saved plain text (using .plain_text attribute): {txt_path.relative_to(output_dir)}")
             else: print(f"INFO (save_text): '.{plain_text_attr}' attribute exists but is empty.")

        if not saved_plain:
             print(f"ERROR (save_text): Could not save plain text for {base_filename}. Failed known methods.")

    except Exception as e:
        print(f"ERROR (save_text): Failed saving plain text for {base_filename}: {e}")

    return counts

# --- Image Saving ---
# (save_images function remains exactly the same as in the previous complete version)
def save_images(document: Any, output_dir: Path, base_filename: str) -> Tuple[int, Dict[int, str]]:
    """Saves images and associated metadata."""
    if Image is None:
        print("ERROR (save_images): Pillow (PIL) not available. Skipping image saving.")
        return 0, {}
    if not document:
        print(f"ERROR (save_images): Document object is None for {base_filename}.")
        return 0, {}

    iterate_items_method = 'iterate_items'
    if not hasattr(document, iterate_items_method):
        print(f"ERROR (save_images): Document lacks '{iterate_items_method}' method.")
        return 0, {}

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_image_paths: Dict[int, str] = {} # {element_index: relative_posix_path}
    saved_count = 0
    processed_image_elements = 0
    get_image_method = 'get_image'
    page_no_attr = 'page_no'
    bbox_attr = 'bbox'
    caption_attr = 'caption'
    label_attr = 'label' # Added for consistency

    try:
        # Iterate using element index (idx)
        for idx, item_tuple in enumerate(getattr(document, iterate_items_method)()):
            element = item_tuple[0] if isinstance(item_tuple, tuple) else item_tuple

            # Check type and label for Picture elements
            element_label = getattr(element, label_attr, None)
            # Convert DocItemLabel enum to string if necessary for comparison
            element_label_str = element_label.value if hasattr(element_label, 'value') else str(element_label)

            if isinstance(element, PictureItem) or element_label_str == DocItemLabel.PICTURE:
                processed_image_elements += 1
                img_filename = f"{base_filename}-img-{saved_count+1}.png"
                meta_filename = f"{base_filename}-img-{saved_count+1}.json"
                img_path = image_dir / img_filename
                meta_path = image_dir / meta_filename
                # Store relative path POSIX style for markdown compatibility
                relative_img_path = (Path("images") / img_filename).as_posix()

                pil_image: Optional[Image.Image] = None
                try:
                    if not hasattr(element, get_image_method):
                        print(f"WARNING (save_images): Image element {idx} lacks '{get_image_method}' method. Skipping.")
                        continue

                    # Attempt calling get_image with and without the document argument
                    try:
                         pil_image = getattr(element, get_image_method)(document)
                    except TypeError as te:
                         # Handle common error if 'document' shouldn't be passed
                         if "takes 1 positional argument but 2 were given" in str(te) or \
                            "takes 0 positional arguments but 1 was given" in str(te): # Handle both cases
                              try:
                                   pil_image = getattr(element, get_image_method)()
                              except Exception as e_retry:
                                   print(f"ERROR (save_images): Calling {get_image_method}() failed after TypeError: {e_retry}")
                                   continue
                         else:
                              raise # Re-raise other TypeErrors

                    if pil_image:
                        pil_image.save(img_path, "PNG")
                        page_num = getattr(element, page_no_attr, 'N/A')
                        bbox = getattr(element, bbox_attr, None)
                        caption = getattr(element, caption_attr, None)

                        # Ensure bbox is serializable (e.g., list of floats/ints)
                        bbox_serializable = None
                        if hasattr(bbox, 'dict'): bbox_serializable = bbox.dict()
                        elif isinstance(bbox, (list, tuple)): bbox_serializable = list(bbox)
                        # Add checks for specific BBox types if needed

                        metadata = {
                            "filename": img_path.name, "page_number": page_num,
                            "position_bbox_pts": bbox_serializable,
                            "size_pixels": pil_image.size if pil_image else None,
                            "caption": caption, "element_index": idx,
                            "element_type": type(element).__name__,
                            "element_label": element_label_str # Store string version
                        }

                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, default=str) # Use default=str for robustness

                        saved_image_paths[idx] = relative_img_path # Store path with forward slashes using element index
                        saved_count += 1
                    else:
                        page_num_warn = getattr(element, page_no_attr, 'N/A')
                        print(f"WARNING (save_images): {get_image_method} returned None for image element {idx} (Page {page_num_warn}).")

                except Exception as e_inner:
                    page_num_err = getattr(element, page_no_attr, 'N/A')
                    print(f"ERROR (save_images): Processing image element {idx} (Page {page_num_err}): {type(e_inner).__name__} - {e_inner}")

    except Exception as e_outer:
        print(f"ERROR (save_images): Iterating items failed for {base_filename}: {e_outer}")
        import traceback; traceback.print_exc() # More detailed error

    if processed_image_elements > 0:
        print(f"  Processed {processed_image_elements} image elements, saved {saved_count} images.")
    elif processed_image_elements == 0:
         print(f"  No image elements found or processed.")

    return saved_count, saved_image_paths


# --- Table Saving ---
# (save_tables function remains exactly the same as in the previous complete version)
def save_tables(document: Any, output_dir: Path, base_filename: str, isOpenpyxlAvailable: bool) -> Tuple[int, Dict[int, str]]:
    """Saves tables as CSV (and optionally XLSX) with metadata."""
    if pd is None:
        print("ERROR (save_tables): pandas not available. Skipping table saving.")
        return 0, {}
    if not document:
        print(f"ERROR (save_tables): Document object is None for {base_filename}.")
        return 0, {}

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    saved_table_paths: Dict[int, str] = {} # {original_table_index: relative_posix_path}
    saved_count = 0
    tables_attr = 'tables'
    iterate_items_method = 'iterate_items' # Keep for fallback/verification
    export_method = 'export_to_dataframe'
    page_no_attr = 'page_no'
    bbox_attr = 'bbox'
    label_attr = 'label' # Added for consistency

    doc_tables_list: Optional[List[Any]] = None
    table_source = "unknown"
    element_to_orig_index_map = {} # Define here for broader scope

    # Strategy 1: Prefer the '.tables' attribute if it exists and is a list
    if hasattr(document, tables_attr):
        tables_data = getattr(document, tables_attr)
        if isinstance(tables_data, list):
             doc_tables_list = tables_data
             table_source = f"document.{tables_attr}"
             print(f"INFO (save_tables): Found {len(doc_tables_list)} tables via '{tables_attr}' attribute.")
        elif tables_data is not None:
             print(f"WARNING (save_tables): '.{tables_attr}' attribute exists but is not a list (type: {type(tables_data)}). Trying iteration.")

    # Strategy 2: If '.tables' was unusable or doesn't exist, find TableItems via iteration
    if doc_tables_list is None and hasattr(document, iterate_items_method):
        print("INFO (save_tables): Finding tables via iteration...")
        iter_tables = []
        try:
            # Store element along with its original iteration index
            elements_with_indices = list(enumerate(getattr(document, iterate_items_method)()))
            for idx, item_tuple in elements_with_indices:
                element = item_tuple[0] if isinstance(item_tuple, tuple) else item_tuple
                # Check type and label
                element_label = getattr(element, label_attr, None)
                element_label_str = element_label.value if hasattr(element_label, 'value') else str(element_label)
                if isinstance(element, TableItem) or element_label_str == DocItemLabel.TABLE:
                    # Store the element itself and its original index from the main iteration
                    iter_tables.append((element, idx))

            if iter_tables:
                 # Keep only the elements for processing, but we now know their original indices
                 doc_tables_list = [item[0] for item in iter_tables]
                 # Create a map from element ID back to original index for metadata
                 element_to_orig_index_map = {id(item[0]): item[1] for item in iter_tables}
                 table_source = "iteration"
                 print(f"  Found {len(doc_tables_list)} table elements via iteration.")
            else:
                 print("  No table elements found via iteration either.")

        except Exception as e_iter:
            print(f"ERROR (save_tables): Iterating items to find tables failed: {e_iter}")
            # Can't proceed if neither method worked
            if not hasattr(document, tables_attr) or not isinstance(getattr(document, tables_attr, None), list):
                 return 0, {}

    if doc_tables_list is None or not doc_tables_list: # Check if list is None or empty
        print(f"  No tables found or extracted for {base_filename}.")
        return 0, {}

    num_tables = len(doc_tables_list)
    print(f"  Processing {num_tables} tables (source: {table_source}).")

    # Use 'table_idx' for the index within the found list (0 to num_tables-1)
    for table_idx, table_element in enumerate(doc_tables_list):
        # Basic check: is it a TableItem?
        if not isinstance(table_element, TableItem):
            print(f"WARNING (save_tables): Item {table_idx} in table list is not a TableItem (Type: {type(table_element)}). Skipping.")
            continue

        # Determine the original element index if found via iteration
        original_element_index: Any = table_idx # Default if source was .tables (use Any to allow string fallback)
        if table_source == "iteration":
             element_id = id(table_element)
             if element_id in element_to_orig_index_map:
                  original_element_index = element_to_orig_index_map[element_id]
             else:
                  print(f"WARNING (save_tables): Could not map iterated table element {table_idx} back to original index.")
                  original_element_index = f"unknown (iter #{table_idx})" # Mark uncertainty

        table_filename_base = f"{base_filename}-table-{table_idx+1}" # Naming based on 1-based index in found list
        table_path_base = table_dir / table_filename_base
        meta_path = table_dir / f"{table_filename_base}.json"
        csv_filename = f"{table_filename_base}.csv"
        relative_csv_path = (Path("tables") / csv_filename).as_posix() # Relative path for linking

        try:
            if not hasattr(table_element, export_method):
                print(f"WARNING (save_tables): Table {table_idx+1} (Elem Idx: {original_element_index}) lacks '{export_method}' method. Skipping.")
                continue

            table_df: Optional[pd.DataFrame] = None
            try:
                 table_df = getattr(table_element, export_method)()
            except Exception as e_export:
                 print(f"ERROR (save_tables): Exporting table {table_idx+1} (Elem Idx: {original_element_index}) failed: {e_export}")
                 continue # Skip this table if export fails

            if table_df is None or table_df.empty:
                print(f"WARNING (save_tables): Table {table_idx+1} (Elem Idx: {original_element_index}) exported empty/None DataFrame. Skipping.")
                continue

            # --- Save CSV ---
            table_path_csv = table_path_base.with_suffix(".csv")
            csv_saved = False
            try:
                table_df.to_csv(table_path_csv, index=False, encoding='utf-8-sig') # Try utf-8-sig first for Excel compatibility
                csv_saved = True
            except Exception as e_csv_sig:
                print(f"WARNING (save_tables): Saving CSV failed with utf-8-sig for table {table_idx+1}: {e_csv_sig}. Trying utf-8.")
                try:
                    table_df.to_csv(table_path_csv, index=False, encoding='utf-8')
                    csv_saved = True
                except Exception as e_csv_utf8:
                    print(f"ERROR (save_tables): Saving CSV failed with utf-8 as well for table {table_idx+1}: {e_csv_utf8}. Skipping CSV.")

            if not csv_saved:
                 continue # Skip metadata and XLSX if CSV failed

            # --- Save XLSX (Optional) ---
            xlsx_saved_path_str = None
            if isOpenpyxlAvailable:
                try:
                    table_path_xlsx = table_path_base.with_suffix(".xlsx")
                    # Consider engine='openpyxl' if needed, though default usually works
                    table_df.to_excel(table_path_xlsx, index=False)
                    xlsx_saved_path_str = table_path_xlsx.name # Store just the filename for metadata
                except Exception as e_xlsx:
                    # Log error but continue, XLSX is optional
                    print(f"ERROR (save_tables): Saving XLSX failed for table {table_idx+1} (CSV was saved): {e_xlsx}")

            # --- Save Metadata ---
            page_num = getattr(table_element, page_no_attr, 'N/A')
            bbox = getattr(table_element, bbox_attr, None)
            element_label = getattr(table_element, label_attr, None)
            element_label_str = element_label.value if hasattr(element_label, 'value') else str(element_label)

            # Serialize bbox
            bbox_serializable = None
            if hasattr(bbox, 'dict'): bbox_serializable = bbox.dict()
            elif isinstance(bbox, (list, tuple)): bbox_serializable = list(bbox)

            metadata = {
                "filename_csv": table_path_csv.name,
                "filename_xlsx": xlsx_saved_path_str, # Will be None if skipped or failed
                "page_number": page_num,
                "position_bbox_pts": bbox_serializable,
                "dimensions": {"rows": table_df.shape[0], "columns": table_df.shape[1]},
                "column_names": list(table_df.columns),
                "table_index_in_doc_list": table_idx, # 0-based index in the list processed (either .tables or iterated)
                "original_element_index": original_element_index, # Index from document.iterate_items() if applicable
                "element_type": type(table_element).__name__,
                "element_label": element_label_str
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Use the original table index (0-based from the list) as the key for linking in MD
            saved_table_paths[table_idx] = relative_csv_path
            saved_count += 1

        except Exception as e:
            print(f"ERROR (save_tables): Unhandled error processing table {table_idx+1} (Elem Idx: {original_element_index}): {type(e).__name__} - {e}")
            import traceback; traceback.print_exc()

    print(f"  Saved {saved_count} tables (CSV {'+ XLSX' if isOpenpyxlAvailable else ''}).")
    return saved_count, saved_table_paths


# --- Metadata Saving ---
# (save_processing_metadata function remains exactly the same as in the previous complete version)
def save_processing_metadata(
    output_dir: Path,
    base_filename: str,
    input_suffix: str,
    status: str,
    start_time: float,
    end_time: float,
    counts: dict, # Expected: {"text_formats":{}, "images":0, "tables":0, "book_like_markdown":0}
    pipeline_config: Dict,
    errors: Optional[List] = None
):
    """Saves metadata about the processing run itself."""
    meta_path = output_dir / f"{base_filename}_processing_metadata.json"
    error_list = []
    if errors:
        for err in errors:
             # Handle Docling ErrorItem objects specifically if possible
            if hasattr(err, 'error_code') and hasattr(err, 'error_message'):
                 code = getattr(err, 'error_code', 'UNKNOWN_CODE')
                 msg = getattr(err, 'error_message', 'Unknown error message')
                 # Ensure code is string (might be enum)
                 error_list.append({"code": str(code), "message": str(msg)})
            # Handle simple dictionary errors
            elif isinstance(err, dict) and "code" in err and "message" in err:
                 error_list.append({"code": str(err["code"]), "message": str(err["message"])})
            # Handle plain string errors
            elif isinstance(err, str):
                 error_list.append({"code": "STRING_ERROR", "message": err})
            # Fallback for other types (like Exceptions)
            else:
                error_list.append({"code": "OTHER_ERROR", "message": str(err)})

    # Ensure counts dictionary has expected keys, provide defaults if missing
    # Note the keys expected from docling_handler
    final_counts = {
        "text_formats": counts.get("text_formats", {"docling_markdown": 0, "json": 0, "plain_text": 0}),
        "images": counts.get("images", 0),
        "tables": counts.get("tables", 0),
        "book_like_markdown": counts.get("book_like_markdown", 0) # Key name updated
    }

    metadata = {
        "source_file": base_filename + input_suffix,
        "processing_status": status,
        "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(start_time)),
        "end_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(end_time)),
        "duration_seconds": round(end_time - start_time, 2),
        "pipeline_configuration": pipeline_config, # Renamed for clarity
        "output_counts": final_counts, # Use the sanitized counts
        "processing_errors_warnings": error_list # Renamed for clarity
    }
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            # Use default=str to handle potential non-serializable types robustly
            json.dump(metadata, f, indent=2, default=str)
        # print(f"  Saved processing metadata: {meta_path.name}") # Optional: uncomment for verbosity
    except Exception as e:
        print(f"ERROR (save_metadata): Failed saving metadata for {base_filename}: {e}")