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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from pydantic import Field
from typing_extensions import Self


@dataclass(frozen=True)
class Flux2BlockQuant:
    """Per-Linear NVFP4 quant plan for one ``Flux2TransformerBlock``.

    Each field is the :class:`QuantConfig` to apply to the corresponding
    Linear, or ``None`` to leave it in BF16. Construct via :meth:`resolve`
    from the block index + the checkpoint's ``nvfp4_layers_bfl`` metadata.
    """

    attn_qkv: QuantConfig | None = None
    """Self-attn ``to_q``/``to_k``/``to_v`` (BFL ``img_attn.qkv``)."""
    attn_out: QuantConfig | None = None
    """Self-attn ``to_out[0]`` (BFL ``img_attn.proj``)."""
    added_attn_qkv: QuantConfig | None = None
    """Added-attn ``add_{q,k,v}_proj`` (BFL ``txt_attn.qkv``)."""
    added_attn_out: QuantConfig | None = None
    """Added-attn ``to_add_out`` (BFL ``txt_attn.proj``)."""
    ff: QuantConfig | None = None
    """Image FF ``ff.linear_{in,out}`` (BFL ``img_mlp.{0,2}``)."""
    ff_context: QuantConfig | None = None
    """Text FF ``ff_context.linear_{in,out}`` (BFL ``txt_mlp.{0,2}``)."""

    @classmethod
    def resolve(
        cls,
        block_idx: int,
        base: QuantConfig | None,
        nvfp4_layers_bfl: frozenset[str],
    ) -> Flux2BlockQuant:
        """Resolve the per-Linear plan for ``double_blocks.{block_idx}``.

        BFL's NVFP4 exports embed ``_quantization_metadata`` listing each
        Linear that was quantized; layers absent from the list stay BF16.
        When that metadata isn't available (non-NVFP4 runs, or legacy
        checkpoints without metadata), fall back to the dev-NVFP4 uniform
        pattern: img-side attn + both MLPs quantized, txt-side attn BF16.
        """
        if base is None:
            return cls()
        if not nvfp4_layers_bfl:
            # Legacy / dev-NVFP4 default: img-side quantized, txt-side BF16.
            return cls(
                attn_qkv=base,
                attn_out=base,
                added_attn_qkv=None,
                added_attn_out=None,
                ff=base,
                ff_context=base,
            )
        prefix = f"double_blocks.{block_idx}"

        def _for(submod: str) -> QuantConfig | None:
            return base if f"{prefix}.{submod}" in nvfp4_layers_bfl else None

        return cls(
            attn_qkv=_for("img_attn.qkv"),
            attn_out=_for("img_attn.proj"),
            added_attn_qkv=_for("txt_attn.qkv"),
            added_attn_out=_for("txt_attn.proj"),
            ff=_for("img_mlp.0"),
            ff_context=_for("txt_mlp.0"),
        )


def _make_nvfp4_config(num_layers: int, num_single_layers: int) -> QuantConfig:
    """Build a QuantConfig for NVFP4 block-scaled quantization.

    Mirrors the modelopt NVFP4 format used by FLUX.2-NVFP4: block size 16
    on the K axis, static per-tensor input scales, FP8 weight scales.
    """
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16),
    )
    all_layers = set(range(num_layers + num_single_layers))
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=all_layers,
        attn_quantized_layers=all_layers,
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.NVFP4,
        # BFL ships scales in 5D TCGEN5 interleaved layout flattened to 2D,
        # but that storage can't be K-sharded by slicing axis 1 (the 5D dims
        # mix rows and K-blocks). The weight adapter deinterleaves to true
        # ``[M, K//16]`` at load time and lets the runtime
        # ``block_scales_interleave`` op rebuild the 5D layout per-shard.
        scales_pre_interleaved=False,
    )


class Flux2Config(MAXModelConfigBase):
    patch_size: int = 1
    in_channels: int = 128
    out_channels: int | None = None
    num_layers: int = 8
    num_single_layers: int = 48
    attention_head_dim: int = 128
    num_attention_heads: int = 48
    joint_attention_dim: int = 15360
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 2000
    eps: float = 1e-6
    guidance_embeds: bool = True
    """If False (Klein/distilled), no guidance embedder weights are expected."""
    dtype: DType = DType.bfloat16
    devices: list[DeviceRef] = Field(default_factory=lambda: [DeviceRef.GPU()])
    """Devices for tensor parallelism. ``len(devices) == 1`` (default) runs
    single-GPU; larger lengths shard the denoiser across the listed devices.
    The first device hosts replicated components and any sub-models that
    stay single-device (text encoder, VAE, etc.).
    """
    quant_config: QuantConfig | None = None
    """NVFP4 quantization config, populated when encoding is float4_e2m1fnx2."""

    nvfp4_layers_bfl: frozenset[str] = Field(default_factory=frozenset)
    """BFL-named layers that the checkpoint tagged ``nvfp4`` in its
    ``_quantization_metadata``, e.g. ``double_blocks.0.img_attn.qkv``.
    Empty for non-NVFP4 runs OR for legacy checkpoints without metadata,
    in which case the model falls back to the dev-NVFP4 uniform pattern
    (img-side attn + all MLPs quantized, txt-side attn BF16).
    """

    @classmethod
    def initialize_from_config(
        cls,
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Self:
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in cls.model_fields
        }
        # For NVFP4, the computation dtype stays bfloat16 (FP4 is only for
        # weights); build the QuantConfig so Linear layers know to use the
        # quantized matmul path.
        quant_config: QuantConfig | None = None
        if encoding == "float4_e2m1fnx2":
            quant_config = _make_nvfp4_config(
                init_dict.get("num_layers", 8),
                init_dict.get("num_single_layers", 48),
            )
        raw_dtype = (
            DType.bfloat16
            if quant_config is not None
            else supported_encoding_dtype(encoding)
        )
        init_dict.update(
            {
                "dtype": raw_dtype,
                "devices": [DeviceRef.from_device(d) for d in devices],
                "quant_config": quant_config,
            }
        )
        return cls(**init_dict)
