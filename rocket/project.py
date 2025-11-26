"""Project management for Rocket engine designs.

This module provides a high-level API for organizing and managing
rocket engine design projects, including designs, analysis results,
and associated data.

Example:
    >>> from rocket.project import Project
    >>> from rocket import EngineInputs
    >>>
    >>> # Create or open a project
    >>> project = Project("./raptor_study")
    >>>
    >>> # Save a design
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> project.save_design(inputs, "baseline_v1", description="Initial design")
    >>>
    >>> # Run analysis and save results
    >>> results = project.run_parametric_study("baseline_v1", vary={...})
    >>> project.save_results("baseline_v1", "chamber_pressure_sweep", results)
    >>>
    >>> # List designs
    >>> for design in project.list_designs():
    ...     print(design)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from beartype import beartype

from rocket.storage import DesignFile, LocalStorage, ResultsFile


# =============================================================================
# Project Metadata
# =============================================================================


@beartype
@dataclass
class ProjectMetadata:
    """Metadata about a rocket design project.

    Attributes:
        name: Project name
        description: Project description
        created_at: When the project was created
        modified_at: When the project was last modified
        author: Project author/owner
        version: Project schema version
        tags: Tags for organization
        custom: Custom metadata fields
    """

    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    author: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
            "custom": self.custom,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ProjectMetadata":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            author=data.get("author", ""),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            custom=data.get("custom", {}),
        )


# =============================================================================
# Project Class
# =============================================================================


@beartype
class Project:
    """A rocket engine design project.

    Provides a high-level API for managing engine designs, running
    analyses, and organizing results.

    Directory structure:
        project_root/
        ├── project.json          # Project metadata
        ├── designs/              # Design files (JSON)
        │   ├── baseline_v1.json
        │   └── high_thrust.json
        └── results/              # Analysis results
            ├── baseline_v1/
            │   ├── parametric_study.json
            │   └── parametric_study.parquet
            └── high_thrust/
                └── thermal_analysis.json

    Example:
        >>> project = Project("./my_project", name="Methalox Engine Study")
        >>> project.save_design(inputs, "baseline")
        >>> designs = project.list_designs()
    """

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        description: str = "",
        author: str = "",
    ) -> None:
        """Create or open a project.

        Args:
            path: Path to project directory
            name: Project name (only used when creating new project)
            description: Project description (only used when creating new project)
            author: Project author (only used when creating new project)
        """
        self.root = Path(path)
        self._storage = LocalStorage(self.root)
        self._metadata_path = self.root / "project.json"

        # Load existing project or create new one
        if self._metadata_path.exists():
            with open(self._metadata_path) as f:
                self._metadata = ProjectMetadata.from_json(f.read())
        else:
            # Create new project
            self._metadata = ProjectMetadata(
                name=name or self.root.name,
                description=description,
                author=author,
            )
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save project metadata."""
        self._metadata.modified_at = datetime.now()
        with open(self._metadata_path, "w") as f:
            f.write(self._metadata.to_json())

    # -------------------------------------------------------------------------
    # Project Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Project name."""
        return self._metadata.name

    @property
    def description(self) -> str:
        """Project description."""
        return self._metadata.description

    @property
    def author(self) -> str:
        """Project author."""
        return self._metadata.author

    @property
    def created_at(self) -> datetime:
        """When the project was created."""
        return self._metadata.created_at

    @property
    def modified_at(self) -> datetime:
        """When the project was last modified."""
        return self._metadata.modified_at

    # -------------------------------------------------------------------------
    # Design Management
    # -------------------------------------------------------------------------

    def save_design(
        self,
        inputs: Any,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save an engine design.

        Args:
            inputs: EngineInputs dataclass
            name: Unique name for this design
            description: Human-readable description
            tags: Optional tags for organization
            metadata: Optional additional metadata

        Returns:
            Path where design was saved
        """
        design = DesignFile.from_inputs(
            inputs,
            name=name,
            description=description,
            tags=tags,
            metadata=metadata,
        )
        path = self._storage.save_design(design)
        self._save_metadata()
        return path

    def load_design(self, name: str, input_class: type) -> Any:
        """Load a design and reconstruct the EngineInputs.

        Args:
            name: Design name
            input_class: The dataclass type to reconstruct (e.g., EngineInputs)

        Returns:
            Reconstructed EngineInputs instance
        """
        design = self._storage.load_design(name)
        return design.to_inputs(input_class)

    def get_design_info(self, name: str) -> DesignFile:
        """Get design metadata without reconstructing inputs.

        Args:
            name: Design name

        Returns:
            DesignFile with metadata and serialized parameters
        """
        return self._storage.load_design(name)

    def list_designs(self) -> list[str]:
        """List all designs in the project."""
        return self._storage.list_designs()

    def delete_design(self, name: str) -> None:
        """Delete a design and all associated results.

        Args:
            name: Design name to delete
        """
        self._storage.delete_design(name)
        self._save_metadata()

    def copy_design(self, source_name: str, dest_name: str) -> Path:
        """Copy a design with a new name.

        Args:
            source_name: Name of design to copy
            dest_name: Name for the copy

        Returns:
            Path to the new design file
        """
        design = self._storage.load_design(source_name)
        design.name = dest_name
        design.created_at = datetime.now()
        design.modified_at = datetime.now()
        design.version = 1
        return self._storage.save_design(design)

    # -------------------------------------------------------------------------
    # Results Management
    # -------------------------------------------------------------------------

    def save_results(
        self,
        design_name: str,
        results_name: str,
        analysis_type: str,
        summary: dict[str, Any],
        data: pl.DataFrame | None = None,
    ) -> Path:
        """Save analysis results for a design.

        Args:
            design_name: Name of the design these results are for
            results_name: Unique name for these results
            analysis_type: Type of analysis (e.g., "parametric", "uncertainty")
            summary: Summary statistics and metadata
            data: Optional large dataset to save as Parquet

        Returns:
            Path where results were saved
        """
        results = ResultsFile(
            name=results_name,
            design_name=design_name,
            analysis_type=analysis_type,
            summary=summary,
        )
        path = self._storage.save_results(results, data)
        self._save_metadata()
        return path

    def load_results(
        self,
        results_name: str,
        design_name: str | None = None,
    ) -> tuple[dict[str, Any], pl.DataFrame | None]:
        """Load analysis results.

        Args:
            results_name: Name of results to load
            design_name: Optional design name to narrow search

        Returns:
            Tuple of (summary dict, optional DataFrame)
        """
        results, data = self._storage.load_results(results_name, design_name)
        return results.summary, data

    def list_results(self, design_name: str | None = None) -> list[str]:
        """List available results.

        Args:
            design_name: If provided, only list results for this design

        Returns:
            List of result names
        """
        return self._storage.list_results(design_name)

    # -------------------------------------------------------------------------
    # Project Information
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Get a text summary of the project."""
        info = self._storage.get_project_info()

        lines = [
            f"Project: {self.name}",
            "=" * 50,
            f"Path: {self.root}",
            f"Author: {self.author or '(not set)'}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Modified: {self.modified_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"Designs: {info['num_designs']}",
            f"Results: {info['num_results']}",
            "",
        ]

        if info["designs"]:
            lines.append("Designs:")
            for design in info["designs"]:
                results = info["results_by_design"].get(design, [])
                lines.append(f"  - {design} ({len(results)} results)")

        if self.description:
            lines.extend(["", "Description:", self.description])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Project('{self.root}', name='{self.name}')"

