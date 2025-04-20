# --- START OF FILE dependencies.py ---

# dependencies.py
"""
Handles checking and optional installation of required Python packages.
"""
import importlib
import subprocess
import sys
import logging
from typing import List

# Setup logger for this module
log = logging.getLogger(__name__)

# Map package names (as used in requirements/pip) to the actual module name to import
# Add more mappings here if needed for other packages
PACKAGE_TO_MODULE_MAP = {
    # Auth/Braille/V2/Voice Exam
    'pillow': 'PIL',
    'docling': 'docling',
    'python-dateutil': 'dateutil',
    'python-dotenv': 'dotenv',
    'jsonschema': 'jsonschema',
    'pygame': 'pygame',
    'speechrecognition': 'speech_recognition',
    'pydub': 'pydub',
    'gtts': 'gtts',
    'requests': 'requests',
    'pandas': 'pandas',
    'openpyxl': 'openpyxl',
    'sounddevice': 'sounddevice',
    'soundfile': 'soundfile',
    'librosa': 'librosa',
    'opencv-python': 'cv2',
    'scikit-learn': 'sklearn',
    'streamlit-mic-recorder': 'streamlit_mic_recorder',
    'numpy': 'numpy', # Often implicitly installed

    # YouTube Q&A
    'pytubefix': 'pytubefix',
    'moviepy': 'moviepy',
    'langchain': 'langchain',
    'langchain-openai': 'langchain_openai', # Correct package for ChatOpenAI
    'openai': 'openai', # Dependency for langchain-openai
    'faiss-cpu': 'faiss', # Or 'faiss-gpu' if GPU is intended
    'sentence-transformers': 'sentence_transformers',
    'torch': 'torch',
    'whisper': 'whisper', # Or 'openai-whisper' if using that package directly
    'langchain-community': 'langchain_community', # Needed for FAISS, HuggingFaceEmbeddings

    # Add others as required by your pipeline components
    # 'deepface': 'deepface' # Optional, handled separately in auth_helpers
}

def install_and_check(packages: List[str], install_if_missing: bool = True) -> bool:
    """
    Checks if packages can be imported, optionally installs missing ones using pip,
    and verifies installation.
    """
    log.info(f"--- Checking Dependencies (Install if missing: {install_if_missing}) ---")
    packages_needing_install = []
    all_available = True

    for package_name_pip in packages:
        package_name_lower = package_name_pip.lower()
        module_name_to_import = PACKAGE_TO_MODULE_MAP.get(package_name_lower, package_name_lower)

        try:
            importlib.import_module(module_name_to_import)
            log.info(f"✅ Found '{package_name_pip}' (imports as '{module_name_to_import}')")
        except ImportError:
            log.warning(f"🟡 Dependency '{package_name_pip}' (imports as '{module_name_to_import}') not found.")
            if install_if_missing:
                packages_needing_install.append(package_name_pip)
            all_available = False
        except Exception as e:
            log.error(f"⚠️ Error while checking for '{package_name_pip}': {e}. Assuming unavailable.")
            if install_if_missing:
                packages_needing_install.append(package_name_pip)
            all_available = False

    if not install_if_missing:
         log.info("--- Dependency Check Complete (Check Only) ---")
         return all_available

    if packages_needing_install:
        packages_to_install_unique = sorted(list(set(packages_needing_install)))
        log.info(f"\nAttempting to install missing packages: {', '.join(packages_to_install_unique)}")
        try:
            pip_command = [
                sys.executable, "-m", "pip", "install", "-q",
                "--disable-pip-version-check",
                # Consider adding --upgrade if needed for specific packages
            ] + packages_to_install_unique

            log.debug(f"Running pip command: {' '.join(pip_command)}")
            subprocess.check_call(pip_command)
            log.info("✅ Installation command executed.")

            log.info("Verifying installation...")
            all_verified = True
            for package_name_pip in packages_to_install_unique:
                package_name_lower = package_name_pip.lower()
                module_name_to_import = PACKAGE_TO_MODULE_MAP.get(package_name_lower, package_name_lower)
                try:
                    # Invalidate caches before trying to import the newly installed module
                    importlib.invalidate_caches()
                    if module_name_to_import in sys.modules:
                         # Reload if already imported (e.g., during previous failed attempt)
                         importlib.reload(sys.modules[module_name_to_import])
                    else:
                         # Import fresh
                         importlib.import_module(module_name_to_import)
                    log.info(f"  ✅ Verified '{package_name_pip}'")
                except ImportError:
                    log.error(f"  ❌ VERIFICATION FAILED: Could not import '{module_name_to_import}' after installing '{package_name_pip}'!")
                    all_verified = False
                except Exception as e:
                    log.error(f"  ❌ VERIFICATION ERROR for '{package_name_pip}': {e}")
                    all_verified = False

            if not all_verified:
                 log.critical("\nOne or more critical dependencies failed verification after installation attempt.")
                 log.critical("Please review pip logs or try installing manually:")
                 log.critical(f"  pip install {' '.join(packages_to_install_unique)}")
                 return False
            else:
                 log.info("✅ All newly installed dependencies verified successfully.")
                 all_available = True

        except subprocess.CalledProcessError as e:
            log.critical(f"\n❌ ERROR: Failed to install packages using pip. Command exited with code {e.returncode}.")
            log.critical("Please review pip logs or try installing manually:")
            log.critical(f"  pip install {' '.join(packages_to_install_unique)}")
            return False
        except Exception as e:
             log.exception(f"\n❌ UNEXPECTED ERROR during installation: {e}")
             return False
    else:
        log.info("All required dependencies were already installed.")
        all_available = True

    log.info("--- Dependency Check Complete ---")
    return all_available

# --- END OF FILE dependencies.py ---