"""Output management for Rocket.

This module provides utilities for organizing outputs from rocket design
analyses into structured directories with consistent naming.

Example:
    >>> from openrocketengine.output import OutputContext
    >>> with OutputContext("my_engine_study") as ctx:
    ...     fig.savefig(ctx.path("engine_dashboard.png"))
    ...     contour.to_csv(ctx.path("nozzle_contour.csv"))
    ...     ctx.save_summary({"isp": 300, "thrust": 50000})
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from beartype import beartype


@beartype
class OutputContext:
    """Context manager for organizing analysis outputs.

    Creates a structured output directory with:
    - Timestamp-based naming for version control
    - Subdirectories for different output types
    - Automatic metadata logging
    - Summary JSON export

    Directory structure:
        {base_dir}/{name}_{timestamp}/
        ├── plots/          # Visualization outputs
        ├── data/           # CSV, contour exports
        ├── reports/        # Text summaries
        └── metadata.json   # Run information

    Attributes:
        name: Study/analysis name
        output_dir: Path to the output directory
        timestamp: Creation timestamp
    """

    def __init__(
        self,
        name: str,
        base_dir: str | Path | None = None,
        include_timestamp: bool = True,
        create_subdirs: bool = True,
    ) -> None:
        """Initialize output context.

        Args:
            name: Name for this analysis/study (used in directory name)
            base_dir: Base directory for outputs. Defaults to ./outputs/
            include_timestamp: Whether to include timestamp in directory name
            create_subdirs: Whether to create plots/, data/, reports/ subdirs
        """
        self.name = name
        self.timestamp = datetime.now()
        self._include_timestamp = include_timestamp
        self._create_subdirs = create_subdirs
        self._metadata: dict[str, Any] = {
            "name": name,
            "created": self.timestamp.isoformat(),
            "files": [],
        }

        # Determine base directory
        if base_dir is None:
            base_dir = Path.cwd() / "outputs"
        self.base_dir = Path(base_dir)

        # Create output directory name
        if include_timestamp:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            dir_name = f"{name}_{timestamp_str}"
        else:
            dir_name = name

        self.output_dir = self.base_dir / dir_name
        self._entered = False

    def __enter__(self) -> "OutputContext":
        """Enter the context and create directories."""
        self._entered = True

        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        if self._create_subdirs:
            (self.output_dir / "plots").mkdir(exist_ok=True)
            (self.output_dir / "data").mkdir(exist_ok=True)
            (self.output_dir / "reports").mkdir(exist_ok=True)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and write metadata."""
        self._metadata["completed"] = datetime.now().isoformat()
        self._metadata["success"] = exc_type is None

        # Write metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        self._entered = False

    def path(self, filename: str, subdir: str | None = None) -> Path:
        """Get full path for an output file.

        Automatically routes files to appropriate subdirectories based on extension.

        Args:
            filename: Name of the output file
            subdir: Optional subdirectory override. If None, auto-routes:
                    - .png, .pdf, .svg → plots/
                    - .csv, .json → data/
                    - .txt, .md → reports/

        Returns:
            Full path to the output file
        """
        if not self._entered:
            raise RuntimeError("OutputContext must be used as a context manager")

        # Auto-route based on extension if subdir not specified
        if subdir is None and self._create_subdirs:
            ext = Path(filename).suffix.lower()
            if ext in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
                subdir = "plots"
            elif ext in {".csv", ".json", ".npy", ".npz"}:
                subdir = "data"
            elif ext in {".txt", ".md", ".rst", ".log"}:
                subdir = "reports"

        if subdir:
            full_path = self.output_dir / subdir / filename
        else:
            full_path = self.output_dir / filename

        # Track file for metadata
        self._metadata["files"].append(str(full_path.relative_to(self.output_dir)))

        return full_path

    def plots_dir(self) -> Path:
        """Get path to plots subdirectory."""
        return self.output_dir / "plots"

    def data_dir(self) -> Path:
        """Get path to data subdirectory."""
        return self.output_dir / "data"

    def reports_dir(self) -> Path:
        """Get path to reports subdirectory."""
        return self.output_dir / "reports"

    def save_summary(self, summary: dict[str, Any], filename: str = "summary.json") -> Path:
        """Save a summary dictionary as JSON.

        Args:
            summary: Dictionary of summary data
            filename: Output filename

        Returns:
            Path to saved file
        """
        path = self.path(filename, subdir="data")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return path

    def save_text(self, text: str, filename: str = "report.txt") -> Path:
        """Save text to a report file.

        Args:
            text: Text content to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        path = self.path(filename, subdir="reports")
        with open(path, "w") as f:
            f.write(text)
        return path

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        self._metadata[key] = value

    def log(self, message: str) -> None:
        """Log a message to both console and log file.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)

        log_path = self.output_dir / "run.log"
        with open(log_path, "a") as f:
            f.write(formatted + "\n")


@beartype
def get_default_output_dir() -> Path:
    """Get the default output directory.

    Returns ./outputs/ in the current working directory.

    Returns:
        Path to default output directory
    """
    return Path.cwd() / "outputs"


@beartype
def list_outputs(base_dir: str | Path | None = None) -> list[Path]:
    """List all output directories.

    Args:
        base_dir: Base directory to search. Defaults to ./outputs/

    Returns:
        List of output directory paths, sorted by modification time (newest first)
    """
    if base_dir is None:
        base_dir = get_default_output_dir()
    base_dir = Path(base_dir)

    if not base_dir.exists():
        return []

    outputs = [d for d in base_dir.iterdir() if d.is_dir()]
    return sorted(outputs, key=lambda p: p.stat().st_mtime, reverse=True)


@beartype
def clean_outputs(base_dir: str | Path | None = None, keep_latest: int = 5) -> int:
    """Clean old output directories, keeping the N most recent.

    Args:
        base_dir: Base directory to clean. Defaults to ./outputs/
        keep_latest: Number of recent outputs to keep

    Returns:
        Number of directories removed
    """
    outputs = list_outputs(base_dir)

    if len(outputs) <= keep_latest:
        return 0

    to_remove = outputs[keep_latest:]
    for output_dir in to_remove:
        shutil.rmtree(output_dir)

    return len(to_remove)

