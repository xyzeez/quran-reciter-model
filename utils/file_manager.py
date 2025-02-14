"""
File and directory management utilities for the Quran Reciter Identification project.
"""

import os
import shutil
from pathlib import Path
from typing import List, Union, Optional
import json
import glob
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FileManager:
    """Centralized file and directory management."""

    def __init__(self, base_dir: Union[str, Path]):
        """Initialize file manager.

        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = Path(base_dir)
        self.required_dirs = {
            'config': self.base_dir / 'config',
            'data': self.base_dir / 'data',
            'datasets': {
                'root': self.base_dir / 'datasets',
                'raw': self.base_dir / 'datasets/raw',
                'processed': self.base_dir / 'datasets/processed'
            },
            'logs': self.base_dir / 'logs',
            'models': self.base_dir / 'models',
            'utils': self.base_dir / 'utils',
            'notebooks': self.base_dir / 'notebooks',
            'tests': self.base_dir / 'tests',
            'cache': self.base_dir / 'cache',
            'runs': self.base_dir / 'runs'
        }

    def setup_directories(self) -> None:
        """Create all required directories if they don't exist."""
        try:
            # Create base directory if it doesn't exist
            self.base_dir.mkdir(exist_ok=True)

            # Create all required directories
            for dir_path in self._get_all_dirs():
                dir_path.mkdir(parents=True, exist_ok=True)

            logger.info("Directory structure setup completed")

        except Exception as e:
            logger.error(f"Failed to setup directories: {str(e)}")
            raise

    def clean_directory(self, dir_type: str, exclude: Optional[List[str]] = None) -> None:
        """Clean a specific directory while preserving specified files/patterns.

        Args:
            dir_type: Type of directory to clean (e.g., 'cache', 'logs')
            exclude: List of patterns to exclude from cleaning
        """
        try:
            dir_path = self._get_dir_path(dir_type)
            if not dir_path.exists():
                return

            exclude = exclude or []
            for item in dir_path.iterdir():
                # Skip if item matches any exclude pattern
                if any(item.match(pattern) for pattern in exclude):
                    continue

                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

            logger.info(f"Cleaned directory: {dir_type}")

        except Exception as e:
            logger.error(f"Failed to clean directory {dir_type}: {str(e)}")
            raise

    def get_latest_run(self) -> Optional[Path]:
        """Get the path to the latest experiment run."""
        try:
            runs_dir = self._get_dir_path('runs')
            if not runs_dir.exists():
                return None

            runs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not runs:
                return None

            return max(runs, key=lambda x: x.stat().st_mtime)

        except Exception as e:
            logger.error(f"Failed to get latest run: {str(e)}")
            raise

    def get_latest_checkpoint(self, run_dir: Optional[Path] = None) -> Optional[Path]:
        """Get the path to the latest model checkpoint.

        Args:
            run_dir: Optional run directory, uses latest if not specified
        """
        try:
            if run_dir is None:
                run_dir = self.get_latest_run()
                if run_dir is None:
                    return None

            checkpoint_dir = run_dir / 'checkpoints'
            if not checkpoint_dir.exists():
                return None

            checkpoints = list(checkpoint_dir.glob('*.pt'))
            if not checkpoints:
                return None

            return max(checkpoints, key=lambda x: x.stat().st_mtime)

        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {str(e)}")
            raise

    def save_metadata(self, metadata: dict, dir_type: str, filename: str) -> None:
        """Save metadata to a JSON file.

        Args:
            metadata: Dictionary containing metadata
            dir_type: Type of directory to save to
            filename: Name of the metadata file
        """
        try:
            dir_path = self._get_dir_path(dir_type)
            file_path = dir_path / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            logger.info(f"Saved metadata to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def load_metadata(self, dir_type: str, filename: str) -> dict:
        """Load metadata from a JSON file.

        Args:
            dir_type: Type of directory to load from
            filename: Name of the metadata file

        Returns:
            Dictionary containing metadata
        """
        try:
            dir_path = self._get_dir_path(dir_type)
            file_path = dir_path / filename

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

    def create_run_directory(self) -> Path:
        """Create a new run directory with timestamp.

        Returns:
            Path to the created run directory
        """
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            run_dir = self._get_dir_path('runs') / f"run_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (run_dir / 'checkpoints').mkdir(exist_ok=True)
            (run_dir / 'logs').mkdir(exist_ok=True)
            (run_dir / 'metrics').mkdir(exist_ok=True)
            (run_dir / 'configs').mkdir(exist_ok=True)

            logger.info(f"Created run directory: {run_dir}")
            return run_dir

        except Exception as e:
            logger.error(f"Failed to create run directory: {str(e)}")
            raise

    def cleanup_old_runs(self, max_runs: int = 5) -> None:
        """Clean up old run directories, keeping only the N most recent.

        Args:
            max_runs: Maximum number of run directories to keep
        """
        try:
            runs_dir = self._get_dir_path('runs')
            if not runs_dir.exists():
                return

            runs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if len(runs) <= max_runs:
                return

            # Sort runs by modification time (newest first)
            runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove older runs
            for run_dir in runs[max_runs:]:
                shutil.rmtree(run_dir)
                logger.info(f"Removed old run directory: {run_dir}")

        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {str(e)}")
            raise

    def _get_dir_path(self, dir_type: str) -> Path:
        """Get path for a specific directory type."""
        if dir_type in self.required_dirs:
            path = self.required_dirs[dir_type]
            return path['root'] if isinstance(path, dict) else path
        raise ValueError(f"Unknown directory type: {dir_type}")

    def _get_all_dirs(self) -> List[Path]:
        """Get list of all required directory paths."""
        dirs = []
        for path in self.required_dirs.values():
            if isinstance(path, dict):
                dirs.extend(path.values())
            else:
                dirs.append(path)
        return dirs
