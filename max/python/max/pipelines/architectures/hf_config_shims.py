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

"""Eagerly-registered HuggingFace ``AutoConfig`` shims.

HuggingFace AutoConfig shims for model_types that the installed version of
transformers does not recognize natively.
"""

from typing import Any

from transformers import AutoConfig, DeepseekV3Config, PretrainedConfig

# Use the native Gemma4Config if available (transformers >= 5.5.0.dev0),
# otherwise fall back to our shim for older versions.
try:
    from transformers import Gemma4Config as Gemma4HFConfig
except ImportError:

    class Gemma4HFConfig(PretrainedConfig):  # type: ignore[no-redef]
        model_type = "gemma4"

        def __init__(
            self,
            vision_config: Any = None,
            text_config: Any = None,
            *args,
            **kwargs,
        ):
            vision_config = vision_config if vision_config is not None else {}
            text_config = text_config if text_config is not None else {}
            self.vision_config = PretrainedConfig(**vision_config)
            self.text_config = PretrainedConfig(**text_config)
            super().__init__(*args, **kwargs)

    try:
        AutoConfig.register("gemma4", Gemma4HFConfig)
    except ValueError:
        pass


class _Gemma4UnifiedHFConfig(Gemma4HFConfig):
    """Config shim for the public "gemma4_unified" model_type (Gemma 4 12B).

    Registered unconditionally: even transformers releases that ship a
    native Gemma4Config may not register the unified model_type.
    """

    model_type = "gemma4_unified"


try:
    AutoConfig.register("gemma4_unified", _Gemma4UnifiedHFConfig)
except ValueError:
    pass


# The gemma4_assistant MTP draft is registered lazily by its arch module,
# too late for the gemma4 MTP recipe's draft config load; register it eagerly.
class _Gemma4AssistantHFConfig(PretrainedConfig):
    model_type = "gemma4_assistant"


try:
    AutoConfig.register("gemma4_assistant", _Gemma4AssistantHFConfig)
except ValueError:
    pass


# Register custom config since "step3p5" is not in the transformers library.
class Step3p5PretrainedConfig(PretrainedConfig):
    """Custom PretrainedConfig for Step-3.5 so AutoConfig.from_pretrained() works.

    This is the primary location for mapping Step-3.5 field names to the
    standard HuggingFace fields that Llama3Config expects.  A subset of these
    aliases is also applied in Step3p5Config._ensure_hf_config_aliases() as a
    fallback when trust_remote_code=True loads the repo's own config class
    instead of this one.
    """

    model_type = "step3p5"

    def __init__(self, **kwargs: object) -> None:
        # >=5.4 requires len(layer_types) == num_hidden_layers, trim MTP tail.
        # >=5.5 reads max_position_embeddings in __post_init__, so defer rope.
        num_layers = kwargs.get("num_hidden_layers")
        layer_types = kwargs.get("layer_types")
        if isinstance(layer_types, list) and isinstance(num_layers, int):
            kwargs["layer_types"] = layer_types[:num_layers]
        deferred_rope = {
            key: kwargs.pop(key)
            for key in ("rope_scaling", "rope_parameters", "rope_theta")
            if key in kwargs
        }
        super().__init__(**kwargs)
        for key, value in deferred_rope.items():
            setattr(self, key, value)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        # Standard field aliases that Llama3Config reads.
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = getattr(self, "num_attention_groups", 8)
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = 1e-5
        if not hasattr(self, "rope_scaling"):
            self.rope_scaling = None
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        # rope_theta may be a per-layer list; preserve it and set scalar.
        # transformers v5 nests rope config inside `rope_parameters`; fall
        # back to that before the 10000 default to avoid silently using a
        # wrong base frequency under v5.
        rope_theta = getattr(self, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(self, "rope_parameters", None)
            if isinstance(rope_params, dict):
                rope_theta = rope_params.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000.0
        self.rope_theta = rope_theta
        if isinstance(rope_theta, list):
            self.per_layer_rope_theta = rope_theta
            self.rope_theta = rope_theta[0] if rope_theta else 10000.0
        elif not hasattr(self, "per_layer_rope_theta"):
            self.per_layer_rope_theta = []


try:
    AutoConfig.register("step3p5", Step3p5PretrainedConfig)
except ValueError:
    pass  # Already registered


class _KimiK2Config(PretrainedConfig):
    """Minimal config for the ``kimi_k2`` model type.

    The Eagle3 draft checkpoint (``nvidia/Kimi-K2.5-Thinking-Eagle3``)
    declares ``model_type: "kimi_k2"`` which is not natively registered
    in transformers, and ships no ``auto_map``.  Registering this stub
    lets ``AutoConfig.from_pretrained`` succeed without a manual JSON
    fallback.
    """

    model_type = "kimi_k2"

    def __init__(
        self, max_position_embeddings: int = 262144, **kwargs: object
    ) -> None:
        # transformers >= 5.12 standardizes RoPE params in ``__post_init__``.
        # For scaling rope types (the draft uses ``yarn``) it eagerly reads
        # ``self.max_position_embeddings`` as the ``setdefault`` fallback for
        # ``original_max_position_embeddings``. ``max_position_embeddings`` is
        # not a declared field on the base config, so on this bare stub it is
        # never set before ``__post_init__`` runs, raising ``AttributeError``.
        # Bind it explicitly (from config.json, with a sane default) before
        # delegating to ``super().__init__``.
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kwargs)


AutoConfig.register("kimi_k2", _KimiK2Config, exist_ok=True)


class DeepseekV32HFConfig(DeepseekV3Config):
    """HuggingFace configuration class for DeepSeek-V3.2 models.

    The ``deepseek_v32`` model type is not natively registered in transformers.
    This subclass of ``DeepseekV3Config`` adds the V3.2-specific fields for
    sparse attention (indexer) and registers itself so that
    ``AutoConfig.from_pretrained`` can load DeepSeek-V3.2 repos.
    """

    model_type = "deepseek_v32"

    def __init__(
        self,
        index_head_dim: int = 128,
        index_n_heads: int = 64,
        index_topk: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk


# Register the config with AutoConfig if not already registered.
# This allows AutoConfig.from_pretrained() to work with deepseek_v32 models.
try:
    AutoConfig.register("deepseek_v32", DeepseekV32HFConfig)
except ValueError:
    # Already registered, which is fine.
    pass


class ExaoneConfig(PretrainedConfig):
    """Local config class for EXAONE 3.5 models.

    The ``exaone`` model type is not natively registered in transformers, and the
    remote ``configuration_exaone.py`` shipped in EXAONE 3.5 HuggingFace repos is
    incompatible with the pinned transformers version.  Registering this minimal
    subclass lets ``AutoConfig.from_pretrained`` load the repo's ``config.json``
    without requiring ``trust_remote_code``.
    """

    model_type = "exaone"

    @property
    def num_hidden_layers(self) -> int:
        """Aliases ``num_layers`` to the standard ``num_hidden_layers`` name."""
        return self.num_layers


try:
    AutoConfig.register("exaone", ExaoneConfig)
except ValueError:
    pass


class LagunaHFConfig(PretrainedConfig):
    """Local config class for poolside's Laguna models (``model_type: laguna``).

    Laguna repos point ``auto_map`` at a remote ``configuration_laguna.py`` that
    is incompatible with the pinned ``huggingface_hub``/``transformers`` (it
    decorates a non-dataclass config with ``@strict`` and uses
    ``auto_docstring``). Registering this minimal subclass lets
    ``AutoConfig.from_pretrained`` load the repo's ``config.json`` directly,
    without ``trust_remote_code`` and without executing the remote config code.
    ``LagunaConfig`` (the MAX config) reads the raw fields off this object
    (``rope_parameters``, ``mlp_layer_types``, ``num_experts``, ``gating``, ...),
    so any field present in ``config.json`` is preserved as an attribute.
    """

    model_type = "laguna"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)


try:
    AutoConfig.register("laguna", LagunaHFConfig)
except ValueError:
    pass


# Register Llama4 config shims if the installed transformers version does not
# include native ``llama4``/``llama4_text`` support.
try:
    from transformers import Llama4Config as _Llama4HFConfig
    from transformers import Llama4TextConfig as _Llama4TextHFConfig
except ImportError:

    class _Llama4TextHFConfig(PretrainedConfig):  # type: ignore[no-redef]
        """Minimal shim for the ``llama4_text`` model type.

        Llama4 stores its text-decoder config under the ``text_config`` key.
        This shim forwards all kwargs so that ``Llama4Config.initialize()``
        can extract every field it needs via ``getattr``.
        """

        model_type = "llama4_text"

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
            # Provide defaults for fields that PretrainedConfig expects.
            if not hasattr(self, "rope_scaling"):
                self.rope_scaling = None

    try:
        AutoConfig.register("llama4_text", _Llama4TextHFConfig)
    except ValueError:
        pass  # Already registered

    class _Llama4HFConfig(PretrainedConfig):  # type: ignore[no-redef]
        """Minimal shim for the top-level ``llama4`` model type.

        ``Llama4ForConditionalGeneration`` config.json files nest the text
        decoder config under a ``text_config`` key.  This shim instantiates
        a ``_Llama4TextHFConfig`` for that sub-config so that
        ``Llama4Config.initialize()`` can find it via
        ``getattr(config, "text_config", config)``.
        """

        model_type = "llama4"
        sub_configs = {"text_config": _Llama4TextHFConfig}

        def __init__(
            self,
            text_config: dict[str, Any] | _Llama4TextHFConfig | None = None,
            **kwargs: Any,
        ) -> None:
            if text_config is None:
                text_config = {}
            if isinstance(text_config, dict):
                text_config = _Llama4TextHFConfig(**text_config)
            self.text_config = text_config
            super().__init__(**kwargs)

    try:
        AutoConfig.register("llama4", _Llama4HFConfig)
    except ValueError:
        pass  # Already registered
