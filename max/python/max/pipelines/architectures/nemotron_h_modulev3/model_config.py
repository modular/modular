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
"""Config for Nemotron-H hybrid Mamba-2, attention and MoE models."""

from __future__ import annotations

import enum
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from max.driver import DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kernels import _moe_sigmoid_gemv_router_unsupported
from max.nn.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.weights import resolve_hf_quant_config
from max.support.math import ceildiv
from transformers import AutoConfig
from typing_extensions import Self

from .quantization import (
    NVFP4_GROUP_SIZE,
    ModuleFormat,
    NemotronHQuantScheme,
    Parallelism,
    linear_parallelism,
    parse_quant_scheme,
)

logger = logging.getLogger("max.pipelines")

ATTN_CACHE_KEY = "attn"
STATE_CACHE_KEY = "state"

MOE_TP_CHANNEL_ALIGNMENT = 64
"""The multiple each device's share of a routed expert's channels is padded
to under tensor parallelism. It keeps the W4A4 scales a whole number of
interleaved atoms, and the K of the down projection a whole number of AMD
grouped matmul tiles."""


def moe_channels_per_device(channels: int, num_devices: int) -> int:
    """Returns the routed expert channels each device holds, padding included.

    Under tensor parallelism each device holds its ``1 / num_devices`` of
    every expert's channels, zero-padded to :data:`MOE_TP_CHANNEL_ALIGNMENT`.
    """
    if num_devices == 1:
        return channels
    return (
        ceildiv(channels // num_devices, MOE_TP_CHANNEL_ALIGNMENT)
        * MOE_TP_CHANNEL_ALIGNMENT
    )


class LayerKind(enum.Enum):
    """The mixer of a Nemotron-H layer."""

    MAMBA = "mamba"
    ATTENTION = "attention"
    MOE = "moe"
    MLP = "mlp"


# transformers main renamed the block types; both spellings name the same
# mixers.
_BLOCK_KINDS: dict[str, LayerKind] = {
    "mamba": LayerKind.MAMBA,
    "linear_attention": LayerKind.MAMBA,
    "attention": LayerKind.ATTENTION,
    "full_attention": LayerKind.ATTENTION,
    "moe": LayerKind.MOE,
    "mlp": LayerKind.MLP,
}

# The only values of these fields this implementation builds.
_REQUIRED: dict[str, object] = {
    "attention_bias": False,
    "mlp_bias": False,
    "mamba_proj_bias": False,
    "use_conv_bias": True,
    "tie_word_embeddings": False,
    "residual_in_fp32": False,
    "mlp_hidden_act": "relu2",
    "mamba_hidden_act": "silu",
    "mamba_ssm_cache_dtype": "float32",
    "n_group": 1,
    "topk_group": 1,
    # A latent MoE is Nemotron-3 Super's, a different model.
    "moe_latent_size": None,
}


def parse_layer_kinds(block_types: Sequence[str]) -> list[LayerKind]:
    """Maps a ``layers_block_type`` list to per-layer mixer kinds.

    Raises:
        ValueError: If a block type is not one of the known spellings.
    """
    kinds = []
    for block_type in block_types:
        kind = _BLOCK_KINDS.get(block_type)
        if kind is None:
            raise ValueError(
                f"unknown Nemotron-H block type {block_type!r}; expected one "
                f"of {sorted(_BLOCK_KINDS)}"
            )
        kinds.append(kind)
    return kinds


def _check_quantized_runs_on(
    device_specs: Sequence[DeviceSpec], quant_scheme: NemotronHQuantScheme
) -> None:
    """Checks that the devices run the quantized matmuls.

    Raises:
        ValueError: If a module is quantized and a device is not SM100. The
            NVFP4 block-scaled matmul is SM100-only, and the FP8 path is
            validated only there.
    """
    if not quant_scheme.quantized:
        return
    for device in load_devices(device_specs):
        if not (
            device.api == "cuda"
            and device.architecture_name.startswith("sm_10")
        ):
            raise ValueError(
                "Nemotron-H runs quantized checkpoints on SM100 GPUs only, "
                f"and {device.architecture_name} is not one. Use the BF16 "
                "checkpoint, such as "
                "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16."
            )


def _runs_fused_router(
    device_specs: Sequence[DeviceSpec],
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
) -> bool:
    """Returns whether the MoE router fuses its gate GEMV into top-k.

    Shapes the fused kernel can't run, such as fewer routed experts than one
    warp, keep the separate float32 gate matmul.
    """
    return all(
        device.api in ("hip", "cuda")
        and _moe_sigmoid_gemv_router_unsupported(
            n_routed_experts=num_experts,
            n_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            warp_size=64 if device.api == "hip" else 32,
        )
        is None
        for device in load_devices(device_specs)
    )


@dataclass(kw_only=True)
class NemotronHConfig(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Configuration for a Nemotron-H hybrid decoder.

    Every layer is a pre-norm residual block around one mixer: Mamba-2,
    NoPE grouped-query attention, a relu2 MoE or a relu2 MLP.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    hidden_size: int
    vocab_size: int
    layer_kinds: list[LayerKind]
    layer_norm_epsilon: float

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    intermediate_size: int

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    routed_scaling_factor: float
    norm_topk_prob: bool

    mamba_num_heads: int
    mamba_head_dim: int
    n_groups: int
    ssm_state_size: int
    conv_kernel: int

    quant_scheme: NemotronHQuantScheme

    dtype: DType
    devices: list[DeviceRef]
    max_seq_len: int
    kv_params: MultiKVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    fused_router: bool = False
    """Whether the MoE router runs its gate GEMV, sigmoid and top-k as one
    fused op instead of a separate float32 matmul."""

    def w4a4_mixers(self) -> frozenset[str]:
        """Returns the MoE mixers whose routed experts are NVFP4, run W4A4."""
        return frozenset(
            mixer
            for mixer in self.mixers(LayerKind.MOE)
            if self.quant_scheme.routed_experts_format(mixer, self.num_experts)
            is ModuleFormat.NVFP4_WEIGHT_ONLY
        )

    def shared_expert_slices(self, mixer: str) -> int:
        """Returns how many routed experts' worth of channels the shared
        expert runs as, inside the routed W4A4 grouped matmul.

        relu2 acts on each channel alone, so the shared expert's channels
        split into slices as wide as a routed expert's, and the shared output
        is the sum of the slices' outputs. Each slice runs as one more expert
        that every token picks at weight 1, instead of a separate W4A4 MLP
        whose quantize and matmul launches cost more at decode than its
        weight bytes save.

        Args:
            mixer: The MoE mixer's checkpoint path.

        Returns:
            The number of slices, or 0 when the shared expert runs on its
            own: when it or the routed experts are not NVFP4, or when its
            channels are not a whole number of routed experts'. Under tensor
            parallelism each slice splits by channel like a routed expert.
        """
        if mixer not in self.w4a4_mixers():
            return 0
        if any(
            self.quant_scheme.format_of(f"{mixer}.shared_experts.{proj}")
            is not ModuleFormat.NVFP4_WEIGHT_ONLY
            for proj in ("up_proj", "down_proj")
        ):
            return 0
        slices, rest = divmod(
            self.moe_shared_expert_intermediate_size,
            self.moe_intermediate_size,
        )
        if rest:
            return 0
        return slices

    def runs_as_routed_experts(self, module: str) -> bool:
        """Returns whether a shared-expert projection runs as routed experts.

        See :meth:`shared_expert_slices`.
        """
        mixer, sep, _ = module.partition(".shared_experts.")
        return bool(sep) and self.shared_expert_slices(mixer) > 0

    def linear_shape(self, module: str) -> tuple[int, int]:
        """Returns a dense linear module's ``(in_dim, out_dim)``.

        Raises:
            ValueError: If ``module`` is not a quantizable dense linear.
        """
        hidden = self.hidden_size
        if module == "lm_head":
            return hidden, self.vocab_size
        if module.endswith(".mixer.in_proj"):
            return hidden, (
                self.mamba_intermediate_size
                + self.conv_dim
                + self.mamba_num_heads
            )
        if module.endswith(".mixer.out_proj"):
            return self.mamba_intermediate_size, hidden
        if ".shared_experts." in module:
            inner = self.moe_shared_expert_intermediate_size
        elif module.endswith((".mixer.up_proj", ".mixer.down_proj")):
            inner = self.intermediate_size
        else:
            raise ValueError(f"no linear shape is known for '{module}'")
        return (
            (hidden, inner) if module.endswith(".up_proj") else (inner, hidden)
        )

    def mixers(self, kind: LayerKind) -> frozenset[str]:
        """Returns the checkpoint names of the mixers of one kind."""
        return frozenset(
            f"backbone.layers.{i}.mixer"
            for i, layer_kind in enumerate(self.layer_kinds)
            if layer_kind is kind
        )

    @property
    def sharded_kv_heads(self) -> int:
        """Returns the KV heads the k and v projections hold.

        With more devices than KV heads, each head repeats once per device
        in its group, so that every device holds one whole head.
        """
        return max(self.num_key_value_heads, len(self.devices))

    @property
    def moe_intermediate_size_per_device(self) -> int:
        """Returns the routed expert channels each device holds, padding
        included."""
        return moe_channels_per_device(
            self.moe_intermediate_size, len(self.devices)
        )

    @property
    def nvfp4_experts_split_by_block(self) -> bool:
        """Returns whether each device's share of the routed expert channels
        is a whole number of NVFP4 blocks.

        The down projection's block scales can only be split between blocks.
        """
        share = self.moe_intermediate_size // len(self.devices)
        return share % NVFP4_GROUP_SIZE == 0

    @property
    def mamba_intermediate_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        return (
            self.mamba_intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
        )

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        """Returns the attention leaf beside the Mamba state.

        The attention layers index the KV cache 0, 1, 2, ... in layer order;
        the Mamba layers do the same in the state leaves.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError("Nemotron-H does not support data parallelism")
        kinds = parse_layer_kinds(huggingface_config.layers_block_type)
        attn = kv_cache_config.to_params(
            # With more devices than KV heads, the k and v projections repeat
            # each head across a group of devices.
            allow_kv_head_replication=True,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=kinds.count(LayerKind.ATTENTION),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )
        hf = huggingface_config
        num_mamba_layers = kinds.count(LayerKind.MAMBA)
        # Each device holds the state of its own share of the Mamba heads.
        n = len(devices)
        state = RecurrentStateParams(
            # Separate leaves because the conv and SSM kernels each index
            # their own uniformly strided pool, at different dtypes.
            # NemotronHBackbone unpacks them in this order.
            regions=(
                RecurrentStateRegion(
                    leaf_id="mamba/conv",
                    num_layers=num_mamba_layers,
                    row_shape=(
                        (
                            hf.mamba_num_heads * hf.mamba_head_dim
                            + 2 * hf.n_groups * hf.ssm_state_size
                        )
                        // n,
                        hf.conv_kernel - 1,
                    ),
                    dtype=DType.bfloat16,
                ),
                RecurrentStateRegion(
                    leaf_id="mamba/ssm",
                    num_layers=num_mamba_layers,
                    row_shape=(
                        hf.mamba_num_heads // n,
                        hf.mamba_head_dim,
                        hf.ssm_state_size,
                    ),
                    dtype=DType.float32,
                ),
            ),
            devices=attn.devices,
            data_parallel_degree=attn.data_parallel_degree,
        )
        return MultiKVCacheParams.from_params(
            {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        hf = model_config.huggingface_config
        if hf is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but it could not be loaded."
            )
        encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        kv_cache_format = model_config.kv_cache.kv_cache_format
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=hf,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype_for_encoding(encoding, kv_cache_format),
        )
        config = cls.from_huggingface(
            hf, kv_params=kv_params, devices=devices, max_seq_len=max_seq_len
        )
        _check_quantized_runs_on(model_config.device_specs, config.quant_scheme)
        config.fused_router = _runs_fused_router(
            model_config.device_specs,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
        )
        hf_quant_config = resolve_hf_quant_config(hf, {}) or {}
        if hf_quant_config.get("kv_cache_scheme") and kv_cache_format is None:
            logger.info(
                "Nemotron-H: the checkpoint declares KV-cache scales, which "
                "are not applied; the KV cache stays BF16. Pass "
                "--kv-cache-format float8_e4m3fn for an unscaled FP8 cache."
            )
        return config

    @classmethod
    def from_huggingface(
        cls,
        hf: AutoConfig,
        *,
        kv_params: MultiKVCacheParams,
        devices: list[DeviceRef],
        max_seq_len: int,
    ) -> Self:
        """Reads the architecture out of a Hugging Face config.

        Raises:
            NotImplementedError: If the config asks for a variant this
                implementation does not build.
        """
        for name, expected in _REQUIRED.items():
            value = getattr(hf, name, expected)
            if value != expected:
                raise NotImplementedError(
                    f"Nemotron-H supports {name}={expected!r}, but the "
                    f"checkpoint sets {value!r}"
                )
        limit = tuple(getattr(hf, "time_step_limit", ()))
        if limit and limit != (0.0, math.inf):
            raise NotImplementedError(
                f"Nemotron-H does not clamp dt, but the checkpoint sets "
                f"time_step_limit={limit}"
            )
        n = len(devices)
        kv_heads = hf.num_key_value_heads
        if kv_heads % n and n % kv_heads:
            raise ValueError(
                f"Nemotron-H gives each device whole KV heads, so {n} devices "
                f"must divide its {kv_heads} KV heads or be a multiple of them"
            )
        for count, what in (
            (hf.num_attention_heads, "attention heads"),
            (hf.moe_intermediate_size, "routed expert channels"),
            (
                hf.moe_shared_expert_intermediate_size,
                "shared expert channels",
            ),
            (hf.mamba_num_heads, "Mamba heads"),
            (hf.n_groups, "Mamba groups"),
        ):
            if count % n:
                raise ValueError(
                    f"Nemotron-H shards its {count} {what} across devices, "
                    f"which {n} devices do not divide"
                )
        config = cls(
            hidden_size=hf.hidden_size,
            vocab_size=hf.vocab_size,
            layer_kinds=parse_layer_kinds(hf.layers_block_type),
            layer_norm_epsilon=hf.layer_norm_epsilon,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hf.head_dim,
            intermediate_size=hf.intermediate_size,
            num_experts=hf.n_routed_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
            moe_intermediate_size=hf.moe_intermediate_size,
            moe_shared_expert_intermediate_size=(
                hf.moe_shared_expert_intermediate_size
            ),
            routed_scaling_factor=hf.routed_scaling_factor,
            norm_topk_prob=hf.norm_topk_prob,
            mamba_num_heads=hf.mamba_num_heads,
            mamba_head_dim=hf.mamba_head_dim,
            n_groups=hf.n_groups,
            ssm_state_size=hf.ssm_state_size,
            conv_kernel=hf.conv_kernel,
            quant_scheme=parse_quant_scheme(resolve_hf_quant_config(hf, {})),
            # Quantization is per module: activations and every unquantized
            # weight stay BF16 whatever the checkpoint's encoding.
            dtype=DType.bfloat16,
            devices=devices,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )
        config.check_quantized_modules()
        return config

    def check_quantized_modules(self) -> None:
        """Checks that every quantized module runs in its stored format.

        Raises:
            NotImplementedError: If a quantized module is built in BF16 only,
                or a mixer's routed experts mix formats.
            ValueError: If each device's share of the NVFP4 routed experts'
                channels is not whole NVFP4 blocks, or an NVFP4 row-parallel
                layer's input does not split into whole 64-column scale
                blocks per device.
        """
        self.quant_scheme.check_dense_modules()
        for mixer in self.mixers(LayerKind.MOE):
            self.quant_scheme.routed_experts_format(mixer, self.num_experts)
        n = len(self.devices)
        if self.w4a4_mixers() and not self.nvfp4_experts_split_by_block:
            raise ValueError(
                f"Nemotron-H cannot run the NVFP4 routed experts of this "
                f"checkpoint on {n} devices: each device's share of their "
                f"{self.moe_intermediate_size} channels is not whole "
                f"{NVFP4_GROUP_SIZE}-channel blocks"
            )
        block = 4 * NVFP4_GROUP_SIZE
        for module, fmt in self.quant_scheme.quantized.items():
            if (
                fmt is not ModuleFormat.NVFP4_WEIGHT_ONLY
                or ".experts." in module
                or self.runs_as_routed_experts(module)
                or linear_parallelism(module) is not Parallelism.ROW
            ):
                continue
            in_dim, _ = self.linear_shape(module)
            if in_dim % (n * block) == 0:
                continue
            # Each device's share of the input must be whole scale blocks.
            counts = [str(d) for d in range(1, n) if in_dim % (d * block) == 0]
            if not counts:
                raise ValueError(
                    f"Nemotron-H cannot run the NVFP4 '{module}' of this "
                    f"checkpoint: its {in_dim} input channels are not whole "
                    f"{block}-channel scale blocks"
                )
            supported = (
                f"{', '.join(counts[:-1])} or {counts[-1]}"
                if len(counts) > 1
                else counts[0]
            )
            raise ValueError(
                f"Nemotron-H runs the NVFP4 '{module}' of this checkpoint on "
                f"{supported} devices, not {n}"
            )
