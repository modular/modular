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
from dataclasses import dataclass, field
from typing import Any

from max.pipelines.lib.config import MAXModelConfig
from max.pipelines.lib.hf_utils import HuggingFaceRepo

logger = logging.getLogger(__name__)


@dataclass
class ModelManifest:
    """Registry mapping semantic role strings to MAXModelConfig instances.

    Each model is identified by a role string (e.g. ``"primary"``,
    ``"draft"``, ``"vae"``, ``"unet"``).  Single-model pipelines use the
    ``"primary"`` key by convention; multi-component pipelines (diffusion,
    speculative decoding) store models under their respective roles.
    """

    models: dict[str, MAXModelConfig] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __getitem__(self, role: str) -> MAXModelConfig:
        """Get model config by role.

        Args:
            role: The semantic role string identifying the model.

        Returns:
            The ``MAXModelConfig`` for the given role.

        Raises:
            KeyError: If the role is not found in the registry.
        """
        if role not in self.models:
            raise KeyError(
                f"Role {role!r} not found in registry. "
                f"Available roles: {list(self.models.keys())}"
            )
        return self.models[role]

    def get(
        self, role: str, default: MAXModelConfig | None = None
    ) -> MAXModelConfig | None:
        """Get model config by role, returning *default* if not found."""
        return self.models.get(role, default)

    def __contains__(self, role: str) -> bool:
        """Check if a role exists in the registry."""
        return role in self.models

    def items(self) -> list[tuple[str, MAXModelConfig]]:
        """Return a snapshot of ``(role, config)`` pairs."""
        return list(self.models.items())

    def __len__(self) -> int:
        return len(self.models)

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
            if role not in self.models:
                raise ValueError(
                    f"Cannot partially update role {role!r}: not found. "
                    f"Available roles: {list(self.models.keys())}. "
                    f"Pass config= to add a new component."
                )
            base = self.models[role]
        else:
            base = config

        updated_config = (
            base.model_copy(update=field_overrides) if field_overrides else base
        )
        new_models = {**self.models, role: updated_config}
        return ModelManifest(models=new_models)

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
        case — use ``from_components()`` to configure each component
        individually.

        For single-model repos, a ``MAXModelConfig`` is constructed from
        *model_path* and any extra *kwargs*, then stored under the
        ``"primary"`` key.

        Args:
            model_path: HuggingFace repo ID or local path to the model.
            revision: Optional HuggingFace repo revision (branch, tag, or
                commit hash).  Defaults to the HuggingFace Hub default.
            **kwargs: Additional keyword arguments forwarded to
                ``MAXModelConfig`` (only valid for single-model repos).

        Returns:
            A new ``ModelManifest``.  For transformers-style models this
            has a single ``"primary"`` entry; for diffusion models it
            contains one entry per component.
        """
        repo_kwargs: dict[str, Any] = {"repo_id": model_path}
        if revision is not None:
            repo_kwargs["revision"] = revision
        repo = HuggingFaceRepo(**repo_kwargs)

        components = cls._discover_diffusers_components(repo, revision)
        if components is not None:
            if kwargs:
                raise ValueError(
                    f"from_model_path() does not support extra keyword "
                    f"arguments for multi-component diffusers pipelines. "
                    f"Use from_components() to configure each component "
                    f"individually. Got: {sorted(kwargs)}"
                )
            return cls(models=components)

        config_kwargs: dict[str, Any] = {"model_path": model_path, **kwargs}
        if revision is not None:
            config_kwargs["huggingface_model_revision"] = revision
        model = MAXModelConfig(**config_kwargs)
        return cls(models={"primary": model})

    @classmethod
    def from_components(
        cls, components: dict[str, MAXModelConfig]
    ) -> ModelManifest:
        """Create a registry from named component models.

        Args:
            components: Mapping of role names to model configurations.

        Returns:
            A new ``ModelManifest``.
        """
        return cls(models=dict(components))

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
    ) -> dict[str, MAXModelConfig] | None:
        """Detect a diffusers repo and expand it into per-component configs.

        Reads ``model_index.json`` from *repo*.  If the file exists, each
        component listed in it gets its own ``MAXModelConfig`` with
        ``subfolder`` set to the component name.

        Args:
            repo: A ``HuggingFaceRepo`` handle (local or remote).
            revision: The user-supplied revision, or ``None`` if the caller
                did not specify one.  Only propagated to each component's
                ``huggingface_model_revision`` when explicitly provided.

        Returns:
            A dict mapping component role names to ``MAXModelConfig``
            instances, or ``None`` if this is not a diffusion pipeline.
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
        for component_name, component_info in model_index.items():
            if component_name.startswith("_"):
                continue
            if not isinstance(component_info, list) or len(component_info) != 2:
                continue
            if not all(isinstance(v, str) and v for v in component_info):
                continue

            config_kwargs: dict[str, Any] = {
                "model_path": repo.repo_id,
                "subfolder": component_name,
            }
            if revision is not None:
                config_kwargs["huggingface_model_revision"] = revision
            components[component_name] = MAXModelConfig(**config_kwargs)

        if not components:
            return None

        logger.debug(
            "Expanded diffusers model %s into components: %s",
            repo.repo_id,
            list(components.keys()),
        )
        return components
