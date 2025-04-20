# markdown_reformatter.py
"""
Uses an LLM to reformat raw Markdown output from Docling into a cleaner,
more structured format suitable for downstream processing.

*** WARNING: This version contains HARDCODED OpenRouter credentials for testing. ***
*** REMOVE before committing or sharing publicly. ***
"""

import os
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time
import re
import traceback
import logging
import sys # Needed for basic logging setup in main

# Setup logger for this module
log = logging.getLogger(__name__)
# Basic configuration if run directly
if __name__ != "__main__":
    # Assume logging is configured by the main orchestrator if imported
    pass
else:
    # Basic logging setup for standalone run
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)-7s] %(module)-20s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


# --- Hardcoded OpenRouter Settings (as requested) ---
# WARNING: Hardcoding sensitive keys is a security risk! REMOVE LATER!
HARDCODED_REFORMAT_MODEL = "qwen/qwen-2.5-coder-32b-instruct:free"
HARDCODED_OPENROUTER_API_KEY = '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'
HARDCODED_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# --- End Hardcoded Settings ---

# --- LLM Interaction Function ---
def call_llm_for_reformatting(prompt: str, text_input: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Calls the configured LLM API (Ollama, OpenRouter, etc.) for Markdown reformatting.
    *** NOTE: This version uses HARDCODED values for OpenRouter model, API key, and URL. ***

    Args:
        prompt: The system prompt describing the task.
        text_input: The raw Markdown content to reformat.
        config: The pipeline configuration dictionary (still used for TIMEOUTS, SITE_URL, APP_NAME).

    Returns:
        The reformatted Markdown string, or None on failure.
    """
    # --- Use Hardcoded Values ---
    model = HARDCODED_REFORMAT_MODEL
    api_key = HARDCODED_OPENROUTER_API_KEY
    api_url = HARDCODED_OPENROUTER_API_URL

    # --- Get other values from config ---
    # Use defaults if config is minimal or missing keys during standalone test
    timeouts_config = config.get('TIMEOUTS', {})
    timeout = timeouts_config.get('llm_reformat', 300) # Default to 300s
    site_url = config.get('YOUR_SITE_URL', None) # For OpenRouter headers
    app_name = config.get('YOUR_APP_NAME', 'MarkdownReformatterTest') # For OpenRouter headers

    # Basic validation
    if not model or not api_url or not api_key:
        log.error("Hardcoded OpenRouter model, API URL, or API key is missing in markdown_reformatter.py.")
        return None

    log.info(f"Sending reformatting request to LLM (Model: {model}, Endpoint: {api_url})")
    log.warning("--> Using HARDCODED OpenRouter credentials in markdown_reformatter.py <--")

    # --- Prepare Headers ---
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" # Directly use the hardcoded key
    }
    # Add OpenRouter specific headers
    if site_url: headers["HTTP-Referer"] = site_url
    if app_name: headers["X-Title"] = app_name

    # --- Prepare Payload (Correct for OpenRouter Chat Completions) ---
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text_input}
        ],
        "stream": False
        # Add provider-specific parameters if needed (e.g., temperature)
        # "temperature": 0.7, # Example
    }

    # --- Make API Call ---
    try:
        response = requests.post(url=api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        log.debug(f"LLM API Raw Response: {result}") # Debug log

        # --- Process Response (OpenRouter Chat Completions Format) ---
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content")
        else:
            # Handle potential error format from OpenRouter if choices are missing
            error_detail = result.get("error", {}).get("message", str(result))
            log.error(f"LLM response missing 'choices'. Error detail: {error_detail}")
            return None

        if content:
            log.info("Successfully received reformatted Markdown from LLM.")
            # Clean up potential markdown code block wrapping from the model
            content = content.strip()
            if content.startswith("```markdown") and content.endswith("```"):
                content = content[len("```markdown"): -len("```")].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[len("```"): -len("```")].strip()
            return content
        else:
            log.error(f"No content found in LLM response message.")
            return None

    except requests.exceptions.Timeout:
        log.error(f"API request timed out after {timeout} seconds.")
        return None
    except requests.exceptions.HTTPError as e:
        log.error(f"API request failed with HTTP status {e.response.status_code}.")
        try:
            # Try to parse JSON error from OpenRouter
            error_data = e.response.json()
            error_msg = error_data.get("error", {}).get("message", e.response.text)
            log.error(f"Response body: {error_msg}")
        except json.JSONDecodeError:
            log.error(f"Response body (non-JSON): {e.response.text}")
        except Exception:
            log.error(f"Could not decode error response body: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        log.exception(f"API request failed: {e}")
        return None
    except json.JSONDecodeError:
        # This might happen if the successful response isn't valid JSON somehow
        log.error(f"Failed to decode successful JSON response from LLM API: {response.text}")
        return None
    except Exception as e:
        log.exception(f"An unexpected error occurred during LLM call: {e}")
        return None


# --- Core Reformatting Function ---
def format_markdown_properly(input_path: Path, output_path: Path, config: Dict[str, Any]) -> bool:
    """
    Reads raw markdown, uses LLM (via config) to reformat into a clean, structured
    standard Markdown format, and saves the result.

    Args:
        input_path: Path to the input markdown file (e.g., book_like.md).
        output_path: Path where the formatted markdown should be saved.
        config: The pipeline configuration dictionary.

    Returns:
        True if formatting and saving were successful, False otherwise.
    """
    if not input_path.is_file():
        log.error(f"Input file not found: {input_path}")
        return False

    log.info(f"Reading content for reformatting from: {input_path.name}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_markdown_content = f.read()
    except Exception as e:
        log.exception(f"Failed to read input file {input_path}")
        return False

    if not raw_markdown_content.strip():
        log.warning(f"Input file {input_path.name} is empty. Writing empty output to {output_path.name}.")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("")
            return True
        except Exception as e:
            log.exception(f"Failed to write empty output file {output_path}")
            return False

    # --- System Prompt ---
    system_prompt = """
You are an expert technical editor specializing in converting raw, messy text extracted from PDF exam papers into clean, well-structured Markdown suitable for automated parsing (like JSON conversion).

Analyze the provided input text, which may contain inconsistencies, page headers/footers, broken lines, and non-standard formatting. Your goal is to completely reformat it into standard Markdown according to these rules:

**MANDATORY ACTIONS:**

1.  **Cleanup:** REMOVE ALL occurrences of headers/footers (like "Name:", "Roll number:") and page numbers (like "Page X of Y").
2.  **Structure Identification:** Identify the main title, instructions, each numbered question, their sub-parts (a, b, c...), and the appendix.
3.  **Merge Broken Lines:** Join lines that belong to the same paragraph but were split during text extraction. Ensure paragraphs flow correctly.
4.  **Standard Headings:**
    *   Use `##` for the main exam title (e.g., `## CS305 Computer Architecture...`).
    *   Use `##` for major sections like `## Instructions` and `## Appendix`.
    *   Use `##` followed by the number for main questions (e.g., `## 1. Short answer questions [4 x 1 = 4 marks]`, `## 2. Implementing a pseudo-instruction [...]`). Ensure the question number is correct and sequential.
5.  **Standard Lists:**
    *   Format instructions as a standard Markdown bulleted list (using `-` or `*`).
    *   Format sub-questions consistently using `(a)`, `(b)`, `(c)` etc., indented under their parent question heading. Ensure the lettering is sequential within each question.
6.  **Numbering/Structure Correction:** Verify and FIX the sequence of main question numbers (1, 2, 3...) and sub-question letters ((a), (b)... within each question). Ensure all parts of a question are grouped under the correct main question heading.
7.  **Preserve Content:**
    *   Keep the original text content of questions, instructions, and paragraphs (after merging broken lines).
    *   Fix obvious, unambiguous single-word spelling errors if you find them, but prioritize structural correctness.
    *   **CRITICAL:** Preserve technical terms, variable names (`$t1`, `$at`), code snippets/examples (like `bcp $t1, $t2, $t3`), formulas, and marks `[...]` exactly as they appear.
    *   **CRITICAL:** Preserve ALL Markdown image links (`![...](...)`) and Markdown tables (`|---|...`) exactly as they appear in the input, ensuring they remain correctly positioned relative to the surrounding text.
    *   Preserve any HTML comments (`<!-- ... -->`).
8.  **Spacing:** Use standard Markdown spacing (e.g., blank lines between paragraphs, before/after headings and lists).

**Output Requirements:**
*   Produce ONLY the final, clean, well-structured Markdown text.
*   Do NOT include any explanations, introductory phrases, or Markdown code fences (```) around the entire output.
"""

    # Call the LLM function, passing the config dictionary (still used for timeouts etc.)
    formatted_markdown = call_llm_for_reformatting(system_prompt, raw_markdown_content, config)

    if formatted_markdown:
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_markdown)
                # Ensure newline at EOF
                if not formatted_markdown.endswith('\n'):
                    f.write('\n')
            log.info(f"Successfully saved formatted Markdown to: {output_path.name}")
            return True
        except Exception as e:
            log.exception(f"Failed to write output file {output_path}")
            return False
    else:
        log.error("Failed to get formatted Markdown from LLM. Output file not saved.")
        return False

# --- Test Block ---
if __name__ == "__main__":
    print("--- Running markdown_reformatter.py Standalone Test ---")

    # Define test file paths (adjust if necessary)
    # Input is the 'book_like' markdown generated by docling_handler
    test_input_path = Path(r"C:\Darsh\Hack36_2\pipeline_output\ell788_maj\ell788_maj_book_like.md")
    # Output for the test run
    test_output_path = Path(r"C:\Darsh\Hack36_2\pipeline_output\ell788_maj\ell788_maj_formatted_properly_TEST.md")

    print(f"Input File: {test_input_path}")
    print(f"Output File: {test_output_path}")

    # Create a minimal config dictionary, primarily for timeouts
    test_config = {
        "TIMEOUTS": {
            "llm_reformat": 300 # Use a reasonable timeout for testing
        },
        # Add YOUR_SITE_URL and YOUR_APP_NAME if needed for OpenRouter headers, otherwise they'll be None
        "YOUR_SITE_URL": "http://localhost:8000/test", # Example
        "YOUR_APP_NAME": "MarkdownReformatterTest"    # Example
    }

    if not test_input_path.exists():
        print(f"ERROR: Test input file not found at '{test_input_path}'. Cannot run test.")
        sys.exit(1)

    # Run the formatting function
    success = format_markdown_properly(test_input_path, test_output_path, test_config)

    if success:
        print(f"--- Test Completed Successfully. Output saved to: {test_output_path} ---")
    else:
        print(f"--- Test Failed. Check logs for errors. ---")

    print("-" * 60)