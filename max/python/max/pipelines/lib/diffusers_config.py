# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Configuration classes for parsing diffusers-style model repositories.

Diffusers pipelines use a multi-folder structure with a model_index.json file
at the root that maps component names to their classes and subfolders. This
module provides utilities for parsing and representing this structure.

Example structure:
    model_repo/
    ├── model_index.json
    ├── scheduler/
    │   └── scheduler_config.json
    ├── text_encoder/
    │   ├── config.json
    │   └── model.safetensors
    ├── transformer/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors
    └── vae/
        ├── config.json
        └── diffusion_pytorch_model.safetensors
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import huggingface_hub
from transformers import AutoConfig

from .hf_utils import HuggingFaceRepo

logger = logging.getLogger("max.pipelines")


@dataclass
class DiffusersComponentConfig:
    """Configuration for a single diffusers pipeline component.

    Represents a component (e.g., VAE, transformer, text_encoder) parsed from
    a diffusers repository's model_index.json and its corresponding subfolder.
    """

    name: str
    """Component name as it appears in model_index.json (e.g., 'vae', 'transformer')."""

    subfolder: Path
    """Path to the component's subfolder relative to repo root."""

    library: str
    """Library the component class belongs to (e.g., 'diffusers', 'transformers')."""

    class_name: str
    """Class name of the component (e.g., 'AutoencoderKL', 'T5EncoderModel')."""

    config_path: Path | None = None
    """Path to the component's config.json file, if it exists."""

    weight_paths: list[Path] = field(default_factory=list)
    """List of safetensor/pytorch weight files for this component."""

    config_dict: HuggingFaceRepo | dict[str, Any] = field(default_factory=dict)
    """Parsed contents of the component's config.json."""

    @classmethod
    def from_subfolder(
        cls,
        name: str,
        subfolder: Path,
        library: str,
        class_name: str,
    ) -> DiffusersComponentConfig:
        """Create a DiffusersComponentConfig by scanning a subfolder.

        Args:
            name: Component name from model_index.json.
            subfolder: Path to the component's subfolder.
            library: Library the component belongs to.
            class_name: Class name of the component.

        Returns:
            A populated DiffusersComponentConfig.
        """
        config_path = None
        config_dict = {}
        weight_paths = []

        if subfolder.exists() and subfolder.is_dir():
            # Look for config file - try config.json first, then scheduler_config.json
            # (schedulers in diffusers use scheduler_config.json)
            possible_config = subfolder / "config.json"
            if not possible_config.exists():
                possible_config = subfolder / "scheduler_config.json"

            if possible_config.exists():
                config_path = possible_config
                try:
                    with open(config_path, encoding="utf-8") as f:
                        config_dict = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to parse config for {name}: {e}")

            # Collect weight files (safetensors preferred, then pytorch)
            safetensor_files = list(subfolder.glob("*.safetensors"))
            if safetensor_files:
                weight_paths = sorted(safetensor_files)
            else:
                # Fallback to pytorch weights
                # TODO: Remove this?
                pytorch_files = list(subfolder.glob("*.bin"))
                weight_paths = sorted(pytorch_files)

        return cls(
            name=name,
            subfolder=subfolder,
            library=library,
            class_name=class_name,
            config_path=config_path,
            weight_paths=weight_paths,
            config_dict=config_dict,
        )

    @property
    def has_weights(self) -> bool:
        """Whether this component has weight files."""
        return len(self.weight_paths) > 0

    @property
    def is_sharded(self) -> bool:
        """Whether the component weights are sharded across multiple files."""
        return len(self.weight_paths) > 1


@dataclass
class DiffusersConfig:
    """Parsed representation of a diffusers repository's model_index.json.

    This class parses the model_index.json file and provides access to all
    pipeline components with their configurations and weight paths.
    """

    pipeline_class: str
    """The diffusers pipeline class name (e.g., 'ZImagePipeline')."""

    diffusers_version: str
    """The diffusers version that created this repository."""

    components: dict[str, DiffusersComponentConfig]
    """Mapping of component names to their configurations."""

    model_path: Path
    """Root path of the model repository."""

    raw_config: dict[str, Any] = field(default_factory=dict)
    """Raw contents of model_index.json."""

    _repo_id: str | None = None
    _revision: str | None = None
    _cache_dir: str | Path | None = None
    _token: str | None = None

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> DiffusersConfig:
        """Parse a local diffusers repository.

        Args:
            model_path: Path to the root of the diffusers model repository.

        Returns:
            A populated DiffusersConfig.

        Raises:
            FileNotFoundError: If model_index.json doesn't exist.
            ValueError: If model_index.json is invalid.
        """
        model_path = Path(model_path)
        model_index_path = model_path / "model_index.json"

        if not model_index_path.exists():
            raise FileNotFoundError(
                f"model_index.json not found at {model_path}. "
                "This doesn't appear to be a diffusers-style repository."
            )

        try:
            with open(model_index_path, encoding="utf-8") as f:
                raw_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid model_index.json at {model_index_path}: {e}"
            ) from e

        return cls._from_config_dict(raw_config, model_path)

    @classmethod
    def from_huggingface_repo(
        cls,
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | None = None,
    ) -> DiffusersConfig:
        """Download and parse model_index.json from a HuggingFace repository.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'Tongyi-MAI/Z-Image-Turbo').
            revision: Git revision (branch, tag, or commit hash). Defaults to 'main'.
            cache_dir: Directory to cache downloaded files.
            token: HuggingFace API token for private repos.

        Returns:
            A populated DiffusersConfig.

        Raises:
            EnvironmentError: If the repository or model_index.json cannot be found.
        """
        try:
            # Download model_index.json
            model_index_file = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename="model_index.json",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as e:
            raise OSError(
                f"Failed to download model_index.json from {repo_id}: {e}"
            ) from e

        # Parse the config
        with open(model_index_file, encoding="utf-8") as f:
            raw_config = json.load(f)

        # Determine the cache directory for this repo
        # The model_index_file path contains the cache structure
        model_path = Path(model_index_file).parent

        config = cls._from_config_dict(raw_config, model_path)

        # For HF repos, we need to mark that subfolders may need downloading
        config._repo_id = repo_id
        config._revision = revision
        config._cache_dir = cache_dir
        config._token = token

        return config

    @classmethod
    def _from_config_dict(
        cls,
        raw_config: dict[str, Any],
        model_path: Path,
    ) -> DiffusersConfig:
        """Create a DiffusersConfig from a parsed model_index.json dict.

        Args:
            raw_config: Parsed model_index.json contents.
            model_path: Root path of the model repository.

        Returns:
            A populated DiffusersConfig.
        """
        pipeline_class = raw_config.get("_class_name", "Unknown")
        diffusers_version = raw_config.get("_diffusers_version", "Unknown")

        components = {}
        for key, value in raw_config.items():
            # Skip metadata fields
            if key.startswith("_"):
                continue

            # Value should be [library, class_name] tuple
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                logger.warning(
                    f"Skipping malformed component entry: {key}={value}"
                )
                continue

            library, class_name = value

            # Skip null components (optional components not present)
            if library is None or class_name is None:
                continue

            subfolder = model_path / key
            components[key] = DiffusersComponentConfig.from_subfolder(
                name=key,
                subfolder=subfolder,
                library=library,
                class_name=class_name,
            )

        return cls(
            pipeline_class=pipeline_class,
            diffusers_version=diffusers_version,
            components=components,
            model_path=model_path,
            raw_config=raw_config,
        )

    def get_weight_paths(self, component_name: str) -> list[Path]:
        """Get all weight file paths for a component.

        Args:
            component_name: Name of the component.

        Returns:
            List of weight file paths, empty if component not found.
        """
        component = self.components.get(component_name)
        if component is None:
            return []
        return component.weight_paths

    @property
    def get_component_config(
        self, component_name: str
    ) -> AutoConfig | dict[str, Any] | None:
        """Convenience property to get AutoConfig for text encoder."""
        return self.components.get(component_name).config_dict

    @property
    def text_encoder_model_repo(self) -> HuggingFaceRepo | None:
        """Convenience property to get text encoder model repo."""
        return HuggingFaceRepo(
            repo_id=self._repo_id,
            subfolder=self.components.get("text_encoder").subfolder,
            revision=self._revision,
            trust_remote_code=self.trust_remote_code,
        )

    def __repr__(self) -> str:
        return (
            f"DiffusersConfig("
            f"pipeline_class={self.pipeline_class!r}, "
            f"components={list(self.components.keys())!r})"
        )
