"""Root entry point for Hugging Face Spaces and Streamlit Cloud deployment."""

import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import and execute the UI
from scholarmatch.ui.app import *
