# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Uniform container for one-to-N MAXModelConfig instances identified by role."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from max.pipelines.lib.config import MAXModelConfig
from max.pipelines.lib.hf_utils import HuggingFaceRepo

logger = logging.getLogger(__name__)


class ModelManifest(dict[str, MAXModelConfig]):
    """Registry mapping semantic role strings to MAXModelConfig instances.

    Each model is identified by a role string (e.g. ``"main"``,
    ``"draft"``, ``"vae"``, ``"unet"``).  Single-model pipelines use the
    ``"main"`` key by convention; multi-component pipelines (diffusion,
    speculative decoding) store models under their respective roles.

    ``ModelManifest`` is a ``dict[str, MAXModelConfig]`` subclass, so
    standard dict operations (``[]``, ``in``, ``len``, ``items``, etc.)
    work directly.

    For diffusion pipelines constructed from ``model_index.json``, the
    ``metadata`` property exposes non-component entries (e.g.
    ``_class_name``, ``_diffusers_version``, ``is_distilled``) as a
    plain dict.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self._resolved: bool = False

    # ------------------------------------------------------------------
    # Dict overrides
    # ------------------------------------------------------------------

    def __getitem__(self, role: str) -> MAXModelConfig:
        try:
            return super().__getitem__(role)
        except KeyError:
            raise KeyError(
                f"{role!r} (available roles: {list(self.keys())})"
            ) from None

    def _check_frozen(self) -> None:
        if self._resolved:
            raise TypeError(
                "ModelManifest is frozen after resolve(). "
                "Use with_override() to create a new manifest."
            )

    def __setitem__(self, key: str, value: MAXModelConfig) -> None:
        self._check_frozen()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._check_frozen()
        super().__delitem__(key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the manifest with new model configs."""
        self._check_frozen()
        super().update(*args, **kwargs)

    def pop(self, *args: Any) -> Any:
        """Remove and return a model config by key."""
        self._check_frozen()
        return super().pop(*args)

    def clear(self) -> None:
        """Remove all model configs from the manifest."""
        self._check_frozen()
        super().clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Non-component entries from ``model_index.json``.

        For diffusion pipelines built via ``from_model_path``, this
        contains every key/value pair from ``model_index.json`` that is
        not a component (e.g. ``_class_name``, ``_diffusers_version``,
        ``is_distilled``).  For non-diffusion manifests, returns an
        empty dict.
        """
        return self._metadata

    @property
    def main_architecture_name(self) -> str:
        """Returns the main architecture class name.

        For non-diffusion models (those with a ``"main"`` key),
        delegates to ``MAXModelConfig.architecture_name`` which returns
        ``architectures[0]`` from the HuggingFace config.

        For diffusion pipelines (no ``"main"`` key), returns
        ``metadata["_class_name"]`` (e.g. ``"FluxPipeline"``).

        Raises:
            ValueError: If the architecture name cannot be determined.
        """
        if "main" in self:
            arch_name = self["main"].architecture_name
            if arch_name:
                return arch_name
            raise ValueError(
                f"Cannot determine architecture name for main model "
                f"{self['main'].model_path!r}: HuggingFace config has "
                f"no 'architectures' field."
            )

        # Diffusion pipeline — use stored metadata from model_index.json.
        if not self:
            raise ValueError(
                "Cannot determine architecture name: manifest is empty."
            )
        class_name = self._metadata.get("_class_name")
        if class_name:
            return class_name
        any_config = next(iter(self.values()))
        raise ValueError(
            f"Cannot determine architecture name for diffusion model "
            f"{any_config.model_path!r}: metadata has no "
            f"'_class_name' field."
        )

    @property
    def total_weights_size(self) -> int:
        """Total weight size in bytes across all components.

        Walks every ``MAXModelConfig`` in the manifest and sums
        ``weights_size()``.  Components with no weight files (e.g.
        schedulers) contribute zero.

        Raises:
            RuntimeError: If the manifest has not been resolved via
                ``resolve()`` first.
        """
        if not self._resolved:
            raise RuntimeError(
                "ModelManifest must be resolved before accessing "
                "total_weights_size. Call resolve() first."
            )
        return sum(config.weights_size() for config in self.values())

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_model_info(self) -> None:
        """Logs model configuration information for every model in the manifest.

        Iterates over each role and delegates to
        ``MAXModelConfig.log_model_info()`` for per-model details.
        """
        logger.info("")
        logger.info("Model Information")
        logger.info("=" * 60)
        for role, config in self.items():
            config.log_model_info(role=role)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> None:
        """Validates and resolves every config in the manifest.

        Delegates to ``MAXModelConfig.resolve()`` for each component.
        """
        for config in self.values():
            config.resolve()
        self._resolved = True

    # ------------------------------------------------------------------
    # Immutable update operations
    # ------------------------------------------------------------------

    def with_override(
        self,
        role: str,
        config: MAXModelConfig | None = None,
        **field_overrides: Any,
    ) -> ModelManifest:
        """Return a new manifest with the given role updated.

        Three usage patterns:

        1. **Partial field update** on an existing component::

               manifest.with_override("transformer",
                   weight_path=[Path("w.safetensors")],
                   quantization_encoding="float4_e2m1fnx2",
               )

        2. **Full replacement or addition** of a component::

               manifest.with_override("draft",
                   config=MAXModelConfig(model_path="org/draft"),
               )

        3. **Add/replace with additional field tweaks**::

               manifest.with_override("draft",
                   config=base_cfg,
                   quantization_encoding="q4_0",
               )

        Args:
            role: The semantic role string identifying the component.
            config: A complete ``MAXModelConfig`` to use as the base.
                When ``None``, the existing config for *role* is used
                (the role must already exist).
            **field_overrides: Individual field values to set on the
                config via ``model_copy(update=...)``.

        Returns:
            A new ``ModelManifest`` — the original is not modified.

        Raises:
            ValueError: If *config* is ``None`` and *role* does not
                exist, or if neither *config* nor *field_overrides*
                are provided.
        """
        if config is None and not field_overrides:
            raise ValueError(
                "with_override() requires either a config or field overrides."
            )

        if config is None:
            if role not in self:
                raise ValueError(
                    f"Cannot partially update role {role!r}: not found. "
                    f"Available roles: {list(self.keys())}. "
                    f"Pass config= to add a new component."
                )
            base = self[role]
        else:
            base = config

        updated_config = (
            base.model_copy(update=field_overrides) if field_overrides else base
        )
        new_models = {**self, role: updated_config}
        return ModelManifest(new_models, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        revision: str | None = None,
        **kwargs: Any,
    ) -> ModelManifest:
        """Create a registry from a single model path.

        Inspects *model_path* for a ``model_index.json`` **before**
        constructing any ``MAXModelConfig``.

        If the model is a diffusion pipeline (has a ``model_index.json``),
        the registry is automatically expanded into per-component
        ``MAXModelConfig`` instances.  Extra *kwargs* are rejected in this
        case — construct the ``ModelManifest`` directly to configure
        each component individually.

        For single-model repos, a ``MAXModelConfig`` is constructed from
        *model_path* and any extra *kwargs*, then stored under the
        ``"main"`` key.

        Args:
            model_path: HuggingFace repo ID or local path to the model.
            revision: Optional HuggingFace repo revision (branch, tag, or
                commit hash).  Defaults to the HuggingFace Hub default.
            **kwargs: Additional keyword arguments forwarded to
                ``MAXModelConfig`` (only valid for single-model repos).

        Returns:
            A new ``ModelManifest``.  For transformers-style models this
            has a single ``"main"`` entry; for diffusion models it
            contains one entry per component.
        """
        repo_kwargs: dict[str, Any] = {"repo_id": model_path}
        if revision is not None:
            repo_kwargs["revision"] = revision
        repo = HuggingFaceRepo(**repo_kwargs)

        result = cls._discover_diffusers_components(repo, revision)
        if result is not None:
            if kwargs:
                raise ValueError(
                    f"from_model_path() does not support extra keyword "
                    f"arguments for multi-component diffusers pipelines. "
                    f"Construct the ModelManifest directly to configure "
                    f"each component individually. Got: {sorted(kwargs)}"
                )
            components, metadata = result
            return cls(components, metadata=metadata)

        config_kwargs: dict[str, Any] = {"model_path": model_path, **kwargs}
        if revision is not None:
            config_kwargs["huggingface_model_revision"] = revision
        model = MAXModelConfig(**config_kwargs)
        return cls({"main": model})

    # ------------------------------------------------------------------
    # Diffusers discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_index(repo: HuggingFaceRepo) -> dict[str, Any] | None:
        """Load ``model_index.json`` from a model repository.

        Args:
            repo: A ``HuggingFaceRepo`` handle (local or remote).

        Returns the parsed JSON dict, or ``None`` if the file does not
        exist.
        """
        if repo.repo_type == "local":
            index_path = os.path.join(repo.repo_id, "model_index.json")
            if not os.path.isfile(index_path):
                return None
            with open(index_path) as f:
                return json.load(f)

        # Remote repo — single hf_hub_download call.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            config_path = hf_hub_download(
                repo_id=repo.repo_id,
                filename="model_index.json",
                revision=repo.revision,
            )
        except EntryNotFoundError:
            return None
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _discover_diffusers_components(
        repo: HuggingFaceRepo,
        revision: str | None = None,
    ) -> tuple[dict[str, MAXModelConfig], dict[str, Any]] | None:
        """Detect a diffusers repo and expand it into per-component configs.

        Reads ``model_index.json`` from *repo*.  If the file exists, each
        component listed in it gets its own ``MAXModelConfig`` with
        ``subfolder`` set to the component name.  Non-component entries
        are returned as metadata.

        Args:
            repo: A ``HuggingFaceRepo`` handle (local or remote).
            revision: The user-supplied revision, or ``None`` if the caller
                did not specify one.  Only propagated to each component's
                ``huggingface_model_revision`` when explicitly provided.

        Returns:
            A ``(components, metadata)`` tuple, or ``None`` if this is
            not a diffusion pipeline.  *components* maps role names to
            ``MAXModelConfig`` instances; *metadata* contains all
            non-component entries from ``model_index.json``.
        """
        try:
            model_index = ModelManifest._load_model_index(repo)
        except json.JSONDecodeError:
            raise
        except Exception:
            logger.info(
                "Could not load model_index.json for %s",
                repo.repo_id,
                exc_info=True,
            )
            return None
        if model_index is None:
            return None

        components: dict[str, MAXModelConfig] = {}
        metadata: dict[str, Any] = {}
        for key, value in model_index.items():
            # A valid component is a 2-element list of non-empty strings.
            if (
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(v, str) and v for v in value)
            ):
                config_kwargs: dict[str, Any] = {
                    "model_path": repo.repo_id,
                    "subfolder": key,
                }
                if revision is not None:
                    config_kwargs["huggingface_model_revision"] = revision
                components[key] = MAXModelConfig(**config_kwargs)
            else:
                metadata[key] = value

        if not components:
            return None

        logger.debug(
            "Expanded diffusers model %s into components: %s",
            repo.repo_id,
            list(components.keys()),
        )
        return components, metadata
