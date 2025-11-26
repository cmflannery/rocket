"""File-based design persistence for Rocket.

This module provides abstractions for saving and loading engine designs,
analysis results, and project data. The design allows users to manage
their own files while providing a clean API that could later support
cloud storage backends.

Example:
    >>> from rocket.storage import LocalStorage, DesignFile
    >>> from rocket import EngineInputs
    >>>
    >>> # Save a design
    >>> storage = LocalStorage("./my_project")
    >>> design = DesignFile.from_inputs(inputs, name="baseline_v1")
    >>> storage.save_design(design)
    >>>
    >>> # Load it back
    >>> loaded = storage.load_design("baseline_v1")
    >>> inputs = loaded.to_inputs()
"""

import json
import shutil
from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl
from beartype import beartype

from rocket.units import Quantity

# =============================================================================
# Serialization Helpers
# =============================================================================


def _serialize_value(value: Any) -> Any:
    """Serialize a value to JSON-compatible format."""
    if isinstance(value, Quantity):
        return {
            "__type__": "Quantity",
            "value": value.value,
            "unit": value.unit,
            "dimension": value.dimension,
        }
    elif isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    elif isinstance(value, Path):
        return {"__type__": "Path", "value": str(value)}
    elif is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": "dataclass",
            "__class__": type(value).__name__,
            **{f.name: _serialize_value(getattr(value, f.name)) for f in fields(value)},
        }
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    else:
        return value


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value from JSON format."""
    if isinstance(value, dict):
        if value.get("__type__") == "Quantity":
            return Quantity(value["value"], value["unit"], value["dimension"])
        elif value.get("__type__") == "datetime":
            return datetime.fromisoformat(value["value"])
        elif value.get("__type__") == "Path":
            return Path(value["value"])
        elif value.get("__type__") == "dataclass":
            # Return as dict - caller can reconstruct the dataclass
            return {k: _deserialize_value(v) for k, v in value.items() if not k.startswith("__")}
        else:
            return {k: _deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    else:
        return value


# =============================================================================
# Design File
# =============================================================================


@beartype
@dataclass
class DesignFile:
    """A serializable engine design file.

    Stores all parameters needed to recreate an engine design,
    along with metadata for tracking and organization.

    Attributes:
        name: Unique name for this design (used as filename)
        description: Human-readable description
        created_at: When this design was created
        modified_at: When this design was last modified
        version: Design version number
        parameters: Dict of design parameters (from EngineInputs)
        tags: Optional list of tags for organization
        metadata: Additional metadata (author, notes, etc.)
    """

    name: str
    parameters: dict[str, Any]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_inputs(
        cls,
        inputs: Any,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DesignFile":
        """Create a DesignFile from EngineInputs.

        Args:
            inputs: EngineInputs dataclass
            name: Name for this design
            description: Optional description
            tags: Optional tags for organization
            metadata: Optional additional metadata

        Returns:
            DesignFile ready to be saved
        """
        if not is_dataclass(inputs):
            raise TypeError("inputs must be a dataclass (e.g., EngineInputs)")

        # Extract parameters from dataclass
        parameters = {}
        for f in fields(inputs):
            value = getattr(inputs, f.name)
            parameters[f.name] = _serialize_value(value)

        return cls(
            name=name,
            parameters=parameters,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )

    def to_inputs(self, input_class: type) -> Any:
        """Reconstruct the EngineInputs from stored parameters.

        Args:
            input_class: The dataclass type to reconstruct (e.g., EngineInputs)

        Returns:
            Reconstructed dataclass instance
        """
        # Deserialize parameters
        params = {k: _deserialize_value(v) for k, v in self.parameters.items()}

        # Build the dataclass
        return input_class(**params)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "version": self.version,
            "tags": self.tags,
            "metadata": _serialize_value(self.metadata),
            "parameters": self.parameters,  # Already serialized
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DesignFile":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            metadata=_deserialize_value(data.get("metadata", {})),
            parameters=data["parameters"],  # Keep serialized form
        )


# =============================================================================
# Results File
# =============================================================================


@beartype
@dataclass
class ResultsFile:
    """A file containing analysis results.

    Can store both summary data (JSON) and large datasets (Parquet).

    Attributes:
        name: Unique name for these results
        design_name: Name of the design these results are for
        analysis_type: Type of analysis (e.g., "parametric", "uncertainty", "thermal")
        created_at: When these results were generated
        summary: Summary statistics and metadata (JSON-serializable)
        data_path: Path to Parquet file with full dataset (optional)
    """

    name: str
    design_name: str
    analysis_type: str
    created_at: datetime = field(default_factory=datetime.now)
    summary: dict[str, Any] = field(default_factory=dict)
    data_path: str | None = None

    def to_json(self) -> str:
        """Serialize summary to JSON."""
        data = {
            "name": self.name,
            "design_name": self.design_name,
            "analysis_type": self.analysis_type,
            "created_at": self.created_at.isoformat(),
            "summary": _serialize_value(self.summary),
            "data_path": self.data_path,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ResultsFile":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            design_name=data["design_name"],
            analysis_type=data["analysis_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            summary=_deserialize_value(data.get("summary", {})),
            data_path=data.get("data_path"),
        )


# =============================================================================
# Storage Backend Protocol
# =============================================================================


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends.

    This abstraction allows swapping between local files, cloud storage,
    or database backends without changing application code.
    """

    @abstractmethod
    def save_design(self, design: DesignFile) -> Path:
        """Save a design file.

        Args:
            design: The design to save

        Returns:
            Path where the design was saved
        """
        ...

    @abstractmethod
    def load_design(self, name: str) -> DesignFile:
        """Load a design by name.

        Args:
            name: Design name (without extension)

        Returns:
            The loaded DesignFile

        Raises:
            FileNotFoundError: If design doesn't exist
        """
        ...

    @abstractmethod
    def list_designs(self) -> list[str]:
        """List all available designs.

        Returns:
            List of design names
        """
        ...

    @abstractmethod
    def delete_design(self, name: str) -> None:
        """Delete a design.

        Args:
            name: Design name to delete
        """
        ...

    @abstractmethod
    def save_results(self, results: ResultsFile, data: pl.DataFrame | None = None) -> Path:
        """Save analysis results.

        Args:
            results: Results metadata
            data: Optional large dataset to save as Parquet

        Returns:
            Path where results were saved
        """
        ...

    @abstractmethod
    def load_results(self, name: str) -> tuple[ResultsFile, pl.DataFrame | None]:
        """Load results by name.

        Args:
            name: Results name

        Returns:
            Tuple of (ResultsFile, optional DataFrame)
        """
        ...

    @abstractmethod
    def list_results(self, design_name: str | None = None) -> list[str]:
        """List available results, optionally filtered by design.

        Args:
            design_name: If provided, only list results for this design

        Returns:
            List of result names
        """
        ...


# =============================================================================
# Local Storage Implementation
# =============================================================================


@beartype
class LocalStorage:
    """File-based storage backend.

    Organizes files in a project directory structure:

        project_root/
        ├── project.json          # Project metadata
        ├── designs/              # Design files (JSON)
        │   ├── baseline_v1.json
        │   └── high_thrust.json
        └── results/              # Analysis results
            ├── baseline_v1/
            │   ├── parametric_study.json
            │   └── parametric_study.parquet
            └── trades/
                └── pressure_sweep.parquet

    Example:
        >>> storage = LocalStorage("./my_rocket_project")
        >>> storage.save_design(design)
        >>> designs = storage.list_designs()
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize local storage.

        Args:
            root: Root directory for the project
        """
        self.root = Path(root)
        self._designs_dir = self.root / "designs"
        self._results_dir = self.root / "results"

        # Create directories if they don't exist
        self._designs_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def save_design(self, design: DesignFile) -> Path:
        """Save a design file as JSON."""
        path = self._designs_dir / f"{design.name}.json"

        # Update modified time
        design.modified_at = datetime.now()

        with open(path, "w") as f:
            f.write(design.to_json())

        return path

    def load_design(self, name: str) -> DesignFile:
        """Load a design by name."""
        path = self._designs_dir / f"{name}.json"

        if not path.exists():
            raise FileNotFoundError(f"Design '{name}' not found at {path}")

        with open(path) as f:
            return DesignFile.from_json(f.read())

    def list_designs(self) -> list[str]:
        """List all available designs."""
        return [p.stem for p in self._designs_dir.glob("*.json")]

    def delete_design(self, name: str) -> None:
        """Delete a design and its associated results."""
        design_path = self._designs_dir / f"{name}.json"
        if design_path.exists():
            design_path.unlink()

        # Also delete results for this design
        results_path = self._results_dir / name
        if results_path.exists():
            shutil.rmtree(results_path)

    def save_results(
        self,
        results: ResultsFile,
        data: pl.DataFrame | None = None,
    ) -> Path:
        """Save analysis results.

        Summary is saved as JSON. Large datasets are saved as Parquet.
        """
        # Create directory for this design's results
        results_dir = self._results_dir / results.design_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save large dataset as Parquet if provided
        if data is not None:
            parquet_path = results_dir / f"{results.name}.parquet"
            data.write_parquet(parquet_path)
            results.data_path = str(parquet_path.relative_to(self.root))

        # Save summary as JSON
        json_path = results_dir / f"{results.name}.json"
        with open(json_path, "w") as f:
            f.write(results.to_json())

        return json_path

    def load_results(self, name: str, design_name: str | None = None) -> tuple[ResultsFile, pl.DataFrame | None]:
        """Load results by name.

        Args:
            name: Results name
            design_name: Design name (required if results are organized by design)

        Returns:
            Tuple of (ResultsFile, optional DataFrame)
        """
        # Search for the results file
        if design_name:
            json_path = self._results_dir / design_name / f"{name}.json"
        else:
            # Search all design directories
            matches = list(self._results_dir.glob(f"*/{name}.json"))
            if not matches:
                raise FileNotFoundError(f"Results '{name}' not found")
            json_path = matches[0]

        if not json_path.exists():
            raise FileNotFoundError(f"Results '{name}' not found at {json_path}")

        with open(json_path) as f:
            results = ResultsFile.from_json(f.read())

        # Load Parquet data if available
        data = None
        if results.data_path:
            parquet_path = self.root / results.data_path
            if parquet_path.exists():
                data = pl.read_parquet(parquet_path)

        return results, data

    def list_results(self, design_name: str | None = None) -> list[str]:
        """List available results."""
        if design_name:
            results_dir = self._results_dir / design_name
            if not results_dir.exists():
                return []
            return [p.stem for p in results_dir.glob("*.json")]
        else:
            # List all results across all designs
            return [p.stem for p in self._results_dir.glob("*/*.json")]

    def get_project_info(self) -> dict[str, Any]:
        """Get summary information about the project."""
        designs = self.list_designs()
        all_results = self.list_results()

        return {
            "root": str(self.root),
            "num_designs": len(designs),
            "num_results": len(all_results),
            "designs": designs,
            "results_by_design": {
                d: self.list_results(d) for d in designs
            },
        }

