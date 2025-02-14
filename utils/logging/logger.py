"""
Logging utility for the Quran Reciter Identification project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import platform
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class PipelineLogger:
    # Define emoji constants with fallbacks
    EMOJIS = {
        'rocket': '🚀' if platform.system() != 'Windows' else '>',
        'warning': '⚠️' if platform.system() != 'Windows' else '!',
        'error': '❌' if platform.system() != 'Windows' else 'X',
        'success': '✅' if platform.system() != 'Windows' else '+',
        'info': 'ℹ️' if platform.system() != 'Windows' else 'i',
        'time': '⏱️' if platform.system() != 'Windows' else 't',
        'data': '📊' if platform.system() != 'Windows' else '#',
        'save': '💾' if platform.system() != 'Windows' else 's',
        'cpu': '🖥️' if platform.system() != 'Windows' else 'C',
        'gpu': '🎮' if platform.system() != 'Windows' else 'G'
    }

    def __init__(self, pipeline_name: str):
        """Initialize logger for a specific pipeline."""
        self.pipeline_name = pipeline_name
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.log_dir = Path(f"logs/{pipeline_name}_{self.timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup console with proper encoding
        self.console = Console(force_terminal=True)
        self.setup_logging()

        # Setup progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

    def setup_logging(self):
        """Setup file and console logging."""
        self.logger = logging.getLogger(self.pipeline_name)
        self.logger.setLevel(logging.INFO)

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            self.log_dir / "pipeline.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )

        # Console handler with rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def start_pipeline(self):
        """Start the pipeline with a header."""
        header = f"{self.EMOJIS['rocket']} Starting {self.pipeline_name.title()} Pipeline"
        self.console.print("\n")
        self.console.print(Panel(
            Text(header, style="bold green"),
            subtitle=f"{self.EMOJIS['time']} Started at {self.timestamp}"
        ))
        self.console.print("\n")
        return self.progress

    def create_task(self, description: str, total: int = None):
        """Create a new progress task."""
        return self.progress.add_task(
            f"[cyan]{description}",
            total=total
        )

    def update_task(self, task_id, advance=1, **kwargs):
        """Update a progress task."""
        self.progress.update(task_id, advance=advance, **kwargs)

    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(f"{self.EMOJIS['info']} {message}")

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(f"{self.EMOJIS['warning']} {message}")

    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(f"{self.EMOJIS['error']} {message}")

    def log_success(self, message: str):
        """Log a success message."""
        self.logger.info(f"{self.EMOJIS['success']} {message}")

    def log_stats(self, stats: dict):
        """Log statistics in a table format."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column(f"{self.EMOJIS['data']} Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in stats.items():
            table.add_row(key, str(value))

        self.console.print("\n")
        self.console.print(Panel(table, title="Statistics"))
        self.console.print("\n")

    def log_system_info(self):
        """Log system resource usage."""
        try:
            import psutil
            import torch

            # System info
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # GPU info
            gpu_info = ""
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_properties(0)
                gpu_info = f"\n{self.EMOJIS['gpu']} GPU: {gpu.name} | Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB/{gpu.total_memory/1e9:.1f}GB"

            info = f"{self.EMOJIS['cpu']} CPU: {cpu_percent}% | RAM: {memory.percent}%{gpu_info}"
            self.console.print(Panel(info, title="System Info"))

        except ImportError:
            self.log_warning("System monitoring requires psutil package")

    def __enter__(self):
        """Context manager enter."""
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.progress.__exit__(exc_type, exc_val, exc_tb)
        if exc_type:
            self.log_error(f"Pipeline failed: {exc_val}")
        else:
            self.log_success("Pipeline completed successfully")
