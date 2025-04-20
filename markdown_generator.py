# markdown_generator.py
"""
Contains functions for generating specialized Markdown formats, like the
'book-like' output with embedded images, tables, and code block formatting.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import necessary Docling item types directly here for isinstance checks
# Adapt based on the actual available classes and error messages.
try:
    from docling.datamodel.document import (
        PictureItem, TableItem, TextItem, ListItem
    )
    # Use docling_core if that's where it resides in your installation
    from docling_core.types.doc import DocItemLabel
    print("INFO (markdown_generator): Successfully imported required Docling types.")
except ImportError as e:
    print(f"ERROR (markdown_generator.py): Failed to import required types from docling/docling_core: {e}")
    print("Ensure 'docling' and 'docling_core' are installed correctly.")
    # Define dummy classes so isinstance checks don't raise NameError, but they won't match
    class PictureItem: pass
    class TableItem: # Dummy class with potential attributes/methods for testing
        def __init__(self, rows=None, caption=None):
            self._rows = rows or [["Header 1", "Header 2"], ["Data 1", "Data 2"]]
            self._caption = caption or "Dummy Table"
        def to_markdown(self) -> str:
            return _generate_markdown_table_from_rows(self._rows) if self._rows else "[Dummy Table - No Data]"
        @property
        def rows(self) -> List[List[str]]:
            return self._rows
        @property
        def caption(self) -> Optional[str]:
            return self._caption
    class TextItem: pass
    class ListItem: pass
    class DocItemLabel: # Dummy enum matching expected values
         PAGE_HEADER = "page_header"
         PAGE_FOOTER = "page_footer"
         SECTION_HEADER = "section_header"
         PARAGRAPH = "paragraph"
         TEXT = "text"
         PICTURE = "picture"
         TABLE = "table"
         LIST_ITEM = "list_item"
    print("WARNING (markdown_generator): Using dummy types due to import failure. Label matching might be affected.")

# --- Helper function to generate Markdown table from rows ---
def _generate_markdown_table_from_rows(rows: List[List[str]]) -> str:
    """Generates a Markdown table string from a list of lists."""
    if not rows:
        return ""
    num_cols = len(rows[0]) if rows else 0
    if num_cols == 0:
        return ""

    md_lines = []
    # Header row
    md_lines.append("| " + " | ".join(str(header) for header in rows[0]) + " |")
    # Separator row
    md_lines.append("|" + "---|"*num_cols)
    # Data rows
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(md_lines)

# --- Integrated Markdown Creation ("Book Like") ---
def create_book_like_markdown(
    document: Any,
    output_dir: Path,
    base_filename: str,
    saved_image_paths: Dict[int, str], # {element_index: relative_posix_path}
    saved_table_info: Dict[int, str]   # {table_index_in_doc_list: relative_posix_path}
) -> bool:
    """
    Generates a structured Markdown file with embedded images, embedded tables,
    linked table source files (CSVs), and formats specific text patterns
    (like code/formulas) as code blocks.
    Outputs to '{base_filename}_book_like.md'.

    Args:
        document: The Docling document object.
        output_dir: The base output directory for the document.
        base_filename: The base name for the output file (without extension).
        saved_image_paths: Dictionary mapping element index to relative image path.
        saved_table_info: Dictionary mapping table list index to relative table path (CSV/Excel).

    Returns:
        True if the Markdown file was successfully created, False otherwise.
    """
    if not document:
        print(f"ERROR (create_book_like_md): Document object is None for {base_filename}.")
        return False

    iterate_items_method = 'iterate_items'
    if not hasattr(document, iterate_items_method):
        print(f"ERROR (create_book_like_md): Document lacks '{iterate_items_method}' method.")
        return False

    integrated_md_path = output_dir / f"{base_filename}_book_like.md"
    md_content = []
    tables_attr = 'tables' # To potentially get table list for mapping
    text_attr = 'text'
    label_attr = 'label'
    level_attr = 'level' # Attribute for heading/list level
    caption_attr = 'caption' # For image alt text / table caption
    prefix_attr = 'prefix'   # For list items
    rows_attr = 'rows'       # Assumed attribute for table data (list of lists)
    to_markdown_method = 'to_markdown' # Assumed method for table markdown generation

    # --- Try to get table list and map elements to their index in that list ---
    doc_tables_list = None
    table_element_to_list_index_map = {}
    if hasattr(document, tables_attr):
         tables_data = getattr(document, tables_attr)
         if isinstance(tables_data, list):
              doc_tables_list = tables_data
              try:
                  # Use object ID as key - reliable if elements are unique objects
                  table_element_to_list_index_map = {id(elem): i for i, elem in enumerate(doc_tables_list)}
                  print(f"INFO (create_book_like_md): Mapped {len(table_element_to_list_index_map)} elements from .tables list for linking.")
              except TypeError:
                   print("WARNING (create_book_like_md): Could not create ID map for table elements in .tables list. Table linking will rely on iteration order.")
         elif tables_data is not None:
              print(f"INFO (create_book_like_md): Document has '{tables_attr}' but it's not a list. Table linking will rely on iteration order.")
    else:
         print("INFO (create_book_like_md): Document lacks '.tables' list. Table linking will rely on iteration order.")
    # --- End Table Mapping ---

    print(f"  Generating 'book-like' markdown: {integrated_md_path.name}...")

    # --- Define Code Block Triggers ---
    CODE_BLOCK_START_TRIGGERS = [
        "the running time in the best case is therefore in o ( n log n )",
        "randomization.",
        "assume random ( l, r )",
        "we get a randomized implementation",
        "average analysis.",
        "implementation with insertion sort.",
        "algorithm", "procedure", "function", "input:", "output:",
        "assume random(", "rsplit(", "split(",
    ]
    FORMULA_MARKER = "<!-- formula-not-decoded -->"
    # --- End Code Block Triggers ---

    table_iteration_counter = 0 # Track tables encountered during iteration

    try:
        last_element_was_list = False
        for idx, item_tuple in enumerate(getattr(document, iterate_items_method)()):
            level = 0
            element = None

            if isinstance(item_tuple, tuple) and len(item_tuple) > 0:
                element = item_tuple[0]
                level_from_tuple = item_tuple[1] if len(item_tuple) > 1 and isinstance(item_tuple[1], int) else None
                level = level_from_tuple if level_from_tuple is not None else getattr(element, level_attr, 0)
            else:
                element = item_tuple
                level = getattr(element, level_attr, 0)

            if element is None: continue

            processed = False
            current_element_markdown = "" # Initialize as empty string
            is_code_block = False
            code_language_hint = ""

            # --- 1. Get Text and Apply Replacements ---
            text = ""
            if hasattr(element, text_attr):
                 raw_text = getattr(element, text_attr, '')
                 if isinstance(raw_text, str):
                      text = raw_text.strip()
                      if text:
                           text = text.replace('/lscript', 'l')
                           text = text.replace('/r', 'r')
                 elif raw_text is not None:
                      text = str(raw_text).strip()
                      if text:
                           text = text.replace('/lscript', 'l')
                           text = text.replace('/r', 'r')

            # --- 2. Determine Element Label ---
            element_label = getattr(element, label_attr, None)
            element_label_str = element_label.value if hasattr(element_label, 'value') else str(element_label)

            # --- 3. Check for Code Block / Formula Triggers ---
            if isinstance(element, (TextItem, ListItem)) or \
               element_label_str in [DocItemLabel.PARAGRAPH, DocItemLabel.TEXT] and text:
                text_lower = text.lower()
                if FORMULA_MARKER in text:
                    is_code_block = True
                    code_language_hint = "formula"
                    text = text.replace(FORMULA_MARKER, "").strip()
                    if not text: is_code_block = False
                    processed = True # Mark as processed even if text becomes empty
                elif any(text_lower.startswith(trigger) for trigger in CODE_BLOCK_START_TRIGGERS):
                    is_code_block = True
                    if "random(" in text_lower or "split(" in text_lower or "algorithm" in text_lower or "procedure" in text_lower:
                         code_language_hint = "pseudocode"
                    elif "running time" in text_lower or "average analysis" in text_lower:
                         code_language_hint = "analysis"
                    else: code_language_hint = "text"
                    processed = True

            # --- 4. Process Element ---
            if is_code_block:
                if text:
                     current_element_markdown = f"``` {code_language_hint}\n{text}\n```"
                last_element_was_list = False
            elif element_label_str == DocItemLabel.SECTION_HEADER:
                if text:
                    heading_level = max(1, level + 1) # Use level attribute if available
                    current_element_markdown = f"{'#' * heading_level} {text}"
                processed = True
                last_element_was_list = False
            elif isinstance(element, ListItem) or element_label_str == DocItemLabel.LIST_ITEM:
                prefix = getattr(element, prefix_attr, '* ')
                indent = "  " * level if level > 0 else ""
                clean_prefix = prefix.strip()
                # Ensure space after common list markers if missing
                if clean_prefix and re.match(r'^(\d+\.|[a-z]\)|\*|\-|\+)$', clean_prefix, re.IGNORECASE) and not prefix.endswith(' '):
                    prefix += ' '
                if text or prefix.strip(): # Add even if text is empty but prefix exists (e.g., empty list item)
                     current_element_markdown = f"{indent}{prefix}{text}"
                     processed = True
                     last_element_was_list = True
                else: # Handle case where element is ListItem but has no prefix/text (should be rare)
                     processed = True
                     last_element_was_list = True # Still part of a list context
            elif isinstance(element, PictureItem) or element_label_str == DocItemLabel.PICTURE:
                if idx in saved_image_paths:
                    relative_path = saved_image_paths[idx]
                    alt_text_raw = getattr(element, caption_attr, f"Image {idx+1}")
                    alt_text = re.sub(r'[\s\[\]"\n\r]+', ' ', str(alt_text_raw)).strip() or f"Image {idx+1}"
                    current_element_markdown = f"![{alt_text}]({relative_path})"
                else:
                    current_element_markdown = f"_[Image Element {idx} - Not Saved/Found]_"
                processed = True
                last_element_was_list = False
            elif isinstance(element, TableItem) or element_label_str == DocItemLabel.TABLE:
                table_md = ""
                # --- Attempt to generate Markdown table from element ---
                if hasattr(element, to_markdown_method):
                    try:
                        table_md = getattr(element, to_markdown_method)()
                        if not isinstance(table_md, str): table_md = "" # Ensure it's a string
                    except Exception as md_err:
                        print(f"WARNING (create_book_like_md): Error calling {to_markdown_method} on table element {idx}: {md_err}")
                        table_md = ""
                if not table_md and hasattr(element, rows_attr):
                     rows_data = getattr(element, rows_attr)
                     if isinstance(rows_data, list):
                         table_md = _generate_markdown_table_from_rows(rows_data)
                     else:
                         print(f"WARNING (create_book_like_md): Table element {idx} has '{rows_attr}' but it's not a list.")

                if not table_md:
                    table_md = f"_[Table Element {idx} - Could not extract Markdown content]_"
                # --- End generate Markdown table ---

                # --- Generate link to external file ---
                link_md = ""
                list_index = -1
                element_id = id(element)
                if element_id in table_element_to_list_index_map:
                    list_index = table_element_to_list_index_map[element_id]
                else:
                     print(f"INFO (create_book_like_md): Table element {idx} (ID: {element_id}) not found in map. Using iteration counter {table_iteration_counter} for linking.")

                target_index_for_link = list_index if list_index != -1 else table_iteration_counter
                table_link_num = target_index_for_link + 1 # Use 1-based index for display
                link_text = f"Table {table_link_num}"

                if target_index_for_link in saved_table_info:
                    relative_path = saved_table_info[target_index_for_link]
                    link_md = f"[{link_text}]({relative_path}) (See linked file)"
                else:
                    link_md = f"[{link_text} - Associated File Not Found/Saved]"
                # --- End generate link ---

                # --- Combine table Markdown and link ---
                current_element_markdown = f"{table_md}\n{link_md}" # Add table first, then link on new line

                table_iteration_counter += 1
                processed = True
                last_element_was_list = False
            elif element_label_str in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                # Usually skip headers/footers in main content, but can add if needed
                # if text:
                #     label_name = "Header" if element_label_str == DocItemLabel.PAGE_HEADER else "Footer"
                #     current_element_markdown = f"_{label_name}: {text}_"
                processed = True # Mark as processed to avoid falling into the 'else'
                last_element_was_list = False
            elif not processed and text: # Default handler for simple text/paragraphs
                 current_element_markdown = text
                 processed = True
                 last_element_was_list = False

            # --- Append Content ---
            if processed and current_element_markdown: # Only append if there's content
                is_current_item_list = isinstance(element, ListItem) or element_label_str == DocItemLabel.LIST_ITEM
                # Add blank line before non-list item if the previous item was a list item
                if not is_current_item_list and last_element_was_list:
                    if md_content and md_content[-1].strip(): # Avoid adding multiple blank lines
                        md_content.append("")

                md_content.append(current_element_markdown)
                # Update last_element_was_list *after* appending
                last_element_was_list = is_current_item_list
            elif processed and not current_element_markdown:
                 # If processed but no content generated (e.g., empty header/footer), reset list flag
                 last_element_was_list = False


        # --- Final Assembly ---
        # Join with double newlines, then clean up excessive newlines
        final_md = "\n\n".join(md_content)
        final_md = re.sub(r'\n{3,}', '\n\n', final_md).strip() # Replace 3+ newlines with 2

        with open(integrated_md_path, "w", encoding="utf-8") as f:
            f.write(final_md)
        print(f"  Successfully generated 'book-like' markdown with embedded tables: {integrated_md_path.name}")
        return True

    except Exception as e:
        print(f"ERROR (create_book_like_md): Failed creating markdown: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()
        return False