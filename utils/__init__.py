"""
Utility module initialization.
"""

from pathlib import Path
from .file_manager import FileManager

# Create singleton instance of FileManager
file_manager = FileManager(Path(__file__).parent.parent)

# Setup project directory structure
file_manager.setup_directories()
