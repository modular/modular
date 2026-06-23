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

"""LoRA Modules."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight
from max.graph.quantization import QuantizationEncoding
from max.nn.lora.interfaces import SupportsLoRA

from ..kernels import (
    sgmv_lora_kernel,
    sgmv_qkv_lora_fused,
    sgmv_qkv_lora_kernel,
    sliced_add,
)
from ..kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from ..layer import Module
from ..linear import Linear
from ..quant_config import QuantConfig
from ..stacked_linear import StackedLinear


class LinearLoRA(Module, SupportsLoRA):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_num_loras: int,
        max_lora_rank: int,
        dtype: DType,
        device: DeviceRef,
        has_lora_bias: bool = False,
        name: str | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
    ):
        """Applies a linear transformation and LoRA to input:

        :math:`y_l = (xA^T) @ B^T`.
        :math:`y = (xW^T + b) + y_l`

        .. code-block:: python

            linear_layer = LinearLoRA(
                in_dim=256,
                out_dim=128,
                max_lora_rank=16,
                max_num_loras=100,
                dtype=dtype.float32,
                device=DeviceRef.GPU(),
                has_bias=True,
                has_lora_bias=True,
                name="lora_linear"
            )

            lora_ids: TensorValue # shape: [max_num_loras,]
            lora_ranks: TensorValue # shape: [max_num_loras,]
            input_row_offsets: TensorValue
            linear_layer.set_lora_batch_info(lora_ids, lora_ranks, input_row_offsets)

            # Input tensor of shape: [batch, ..., 256]
            input_tensor: TensorValue
            output = linear_layer(input_tensor)
        """
        super().__init__()

        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.lora_A = Weight(
            name=f"{name}.lora_A.weight" if name else "lora_A.weight",
            dtype=dtype,
            shape=[max_num_loras, max_lora_rank, in_dim],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )
        self.lora_B = Weight(
            name=f"{name}.lora_B.weight" if name else "lora_B.weight",
            dtype=dtype,
            shape=[max_num_loras, out_dim, max_lora_rank],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )
        self.lora_bias = (
            Weight(
                name=f"{name}.lora.bias" if name else "lora.bias",
                dtype=dtype,
                shape=[max_num_loras, out_dim],
                device=device,
                quantization_encoding=quantization_encoding,
                _has_alias=True,
            )
            if has_lora_bias
            else None
        )
        self.lora_ids: TensorValue | None = None
        self.lora_ranks: TensorValue | None = None
        self.num_active_loras: TensorValue | None = None
        self.lora_end_idx: TensorValue | None = None
        self.batch_seq_len: TensorValue | None = None
        self.lora_grouped_offsets: TensorValue | None = None
        self.lora_ids_kv: TensorValue | None = None
        self.lora_grouped_offsets_kv: TensorValue | None = None

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
        num_active_loras: TensorValue,
        lora_end_idx: TensorValue,
        batch_seq_len: TensorValue,
        lora_ids_kv: TensorValue,
        lora_grouped_offsets_kv: TensorValue,
    ) -> None:
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets
        self.num_active_loras = num_active_loras
        self.lora_end_idx = lora_end_idx
        self.batch_seq_len = batch_seq_len
        self.lora_ids_kv = lora_ids_kv
        self.lora_grouped_offsets_kv = lora_grouped_offsets_kv

    def __call__(self, x: TensorValue, y: TensorValue) -> TensorValue:
        if (
            self.lora_ids is None
            or self.lora_ranks is None
            or self.lora_grouped_offsets is None
            or self.num_active_loras is None
            or self.lora_end_idx is None
            or self.batch_seq_len is None
        ):
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        y_lora = sgmv_lora_kernel(
            input=x,
            lora_a=self.lora_A,
            lora_b=self.lora_B,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            grouped_row_offsets=self.lora_grouped_offsets,
            max_lora_seq_len=self.in_dim,
            lora_end_idx=self.lora_end_idx,
            bias=self.lora_bias,
        )

        y = sliced_add(y, y_lora, self.lora_end_idx)

        return y


class QKVLinearLoRA(Module, SupportsLoRA):
    def __init__(
        self,
        in_dim: int,
        q_dim: int,
        kv_dim: int,
        max_num_loras: int,
        max_lora_rank: int,
        dtype: DType,
        device: DeviceRef,
        name: str | None = None,
        quantization_encoding: QuantizationEncoding | None = None,
    ):
        super().__init__()

        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.in_dim = in_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim

        self.lora_A = Weight(
            name=f"{name}.lora_A.weight" if name else "lora_A.weight",
            dtype=dtype,
            shape=[max_num_loras, 3 * max_lora_rank, in_dim],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )

        self.lora_B_q = Weight(
            name=f"{name}.lora_B_q.weight" if name else "lora_B_q.weight",
            dtype=dtype,
            shape=[max_num_loras, q_dim, max_lora_rank],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )
        self.lora_B_kv = Weight(
            name=f"{name}.lora_B_kv.weight" if name else "lora_B_kv.weight",
            dtype=dtype,
            shape=[2 * max_num_loras, kv_dim, max_lora_rank],
            device=device,
            quantization_encoding=quantization_encoding,
            _has_alias=True,
        )

        self.lora_ids: TensorValue | None = None
        self.lora_ranks: TensorValue | None = None
        self.num_active_loras: TensorValue | None = None
        self.lora_end_idx: TensorValue | None = None
        self.batch_seq_len: TensorValue | None = None
        self.lora_grouped_offsets: TensorValue | None = None
        self.lora_ids_kv: TensorValue | None = None
        self.lora_grouped_offsets_kv: TensorValue | None = None

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
        num_active_loras: TensorValue,
        lora_end_idx: TensorValue,
        batch_seq_len: TensorValue,
        lora_ids_kv: TensorValue,
        lora_grouped_offsets_kv: TensorValue,
    ) -> None:
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets
        self.num_active_loras = num_active_loras
        self.lora_end_idx = lora_end_idx
        self.batch_seq_len = batch_seq_len
        self.lora_ids_kv = lora_ids_kv
        self.lora_grouped_offsets_kv = lora_grouped_offsets_kv

    def __call__(
        self,
        x: TensorValue,
        xq: TensorValue,
        kv_collection: PagedCacheValues,
        kv_params: KVCacheParams,
        input_row_offsets: TensorValue,
        layer_idx: TensorValue,
        max_seq_len: int,
    ) -> TensorValue:
        """Computes fused query, key, and value LoRAs with ragged input.

        Args:
            x: The input tensor of shape [total_tokens, hidden_dim].
            qkv_loras: List of 3 LinearLoRA modules for Q, K, and V projections.
            input_row_offsets: 1D tensor indicating the start index of each sequence in `x`.
            kv_collection:
                The key/value cache collection structure.
            layer_idx: Index of the current transformer layer (used for caching).

        Returns:
            TensorValue: The query projections.

        Raises:
            ValueError: If 'set_lora_batch_info' has not been called on the LoRAs.
        """
        if (
            self.lora_ids is None
            or self.lora_ranks is None
            or self.lora_grouped_offsets is None
            or self.num_active_loras is None
            or self.lora_end_idx is None
            or self.batch_seq_len is None
            or self.lora_ids_kv is None
            or self.lora_grouped_offsets_kv is None
        ):
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        xq_lora = sgmv_qkv_lora_kernel(
            input=x,
            lora_a=self.lora_A,
            lora_b_q=self.lora_B_q,
            lora_b_kv=self.lora_B_kv,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            input_row_offsets=input_row_offsets,
            lora_grouped_offsets=self.lora_grouped_offsets,
            lora_end_idx=self.lora_end_idx,
            batch_seq_len=self.batch_seq_len,
            lora_ids_kv=self.lora_ids_kv,
            lora_grouped_offsets_kv=self.lora_grouped_offsets_kv,
            kv_collection=kv_collection,
            kv_params=kv_params,
            layer_idx=layer_idx,
            max_lora_seq_len=max_seq_len,
            max_rank=self.max_lora_rank,
            bias=None,
        )

        xq = sliced_add(xq, xq_lora, self.lora_end_idx)

        return xq


class LoRAMixin(SupportsLoRA):
    """Shared per-batch LoRA state for projection layers.

    Holds the batch metadata the scheduler sets each step via
    :meth:`set_lora_batch_info` and exposes it to the LoRA compute. Shared by
    the plain (:class:`LoRALinear`) and fused-QKV (``StackedLinearLoRA``)
    projections so the plumbing lives in one place; each projection adds its
    own LoRA weights and ``lora()`` / ``qkv_lora()`` compute.
    """

    lora_ids: TensorValue | None = None
    lora_ranks: TensorValue | None = None
    lora_grouped_offsets: TensorValue | None = None
    num_active_loras: TensorValue | None = None
    lora_end_idx: TensorValue | None = None
    batch_seq_len: TensorValue | None = None
    lora_ids_kv: TensorValue | None = None
    lora_grouped_offsets_kv: TensorValue | None = None

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
        lora_grouped_offsets: TensorValue,
        num_active_loras: TensorValue,
        lora_end_idx: TensorValue,
        batch_seq_len: TensorValue,
        lora_ids_kv: TensorValue,
        lora_grouped_offsets_kv: TensorValue,
    ) -> None:
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets
        self.num_active_loras = num_active_loras
        self.lora_end_idx = lora_end_idx
        self.batch_seq_len = batch_seq_len
        self.lora_ids_kv = lora_ids_kv
        self.lora_grouped_offsets_kv = lora_grouped_offsets_kv

    @classmethod
    def from_base(
        cls,
        base: Module,
        max_num_loras: int,
        max_lora_rank: int,
        max_lora_seq_len: int,
    ) -> LoRAMixin:
        """Builds a LoRA-wrapped projection mirroring ``base``.

        Each concrete projection implements this to reuse ``base``'s dims,
        dtype, quantization, and device so its output is unchanged once the
        base weights load, then adds its own adapter weights.
        """
        raise NotImplementedError()


class LoRALinear(Linear, LoRAMixin):
    """A :class:`~max.nn.Linear` projection with an additive LoRA term.

    Drop-in replacement for a base ``Linear``: ``__call__(x)`` returns the base
    projection plus the active LoRA contribution. The base weight keeps its
    ``weight`` name and the adapter weights are ``lora_A`` / ``lora_B``, so an
    wrapped ``o_proj`` exposes ``o_proj.weight`` and ``o_proj.lora_A``,
    matching PEFT-exported checkpoints.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        max_num_loras: int,
        max_lora_rank: int,
        max_lora_seq_len: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quant_config: QuantConfig | None = None,
        name: str | None = None,
        clip_weight: float | None = None,
        has_lora_bias: bool = False,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
            quantization_encoding=quantization_encoding,
            quant_config=quant_config,
            name=name,
            clip_weight=clip_weight,
        )
        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.max_lora_seq_len = max_lora_seq_len
        self.in_dim = in_dim

        # LoRA weights stay in a compute dtype even when the base is quantized.
        lora_dtype = DType.bfloat16 if quant_config is not None else dtype
        self.lora_A = Weight(
            name="lora_A.weight",
            dtype=lora_dtype,
            shape=[max_num_loras, max_lora_rank, in_dim],
            device=device,
            _has_alias=True,
        )
        self.lora_B = Weight(
            name="lora_B.weight",
            dtype=lora_dtype,
            shape=[max_num_loras, out_dim, max_lora_rank],
            device=device,
            _has_alias=True,
        )
        self.lora_bias = (
            Weight(
                name="lora.bias",
                dtype=lora_dtype,
                shape=[max_num_loras, out_dim],
                device=device,
                _has_alias=True,
            )
            if has_lora_bias
            else None
        )

    @classmethod
    def from_base(
        cls,
        base: Module,
        max_num_loras: int,
        max_lora_rank: int,
        max_lora_seq_len: int,
        *,
        has_lora_bias: bool = False,
    ) -> LoRALinear:
        """Builds a :class:`LoRALinear` mirroring an existing ``Linear``.

        The base projection's configuration (dims, dtype, quantization, bias,
        device) is reused so the base output is unchanged once the original
        ``weight`` is loaded into the result.
        """
        assert isinstance(base, Linear)
        if base.quant_config is not None and base.quant_config.is_fp4:
            raise NotImplementedError(
                "LoRA is not yet supported for fp4-quantized Linear layers."
            )
        out_dim, in_dim = (int(dim) for dim in base.weight.shape.static_dims)
        return cls(
            in_dim=in_dim,
            out_dim=out_dim,
            max_num_loras=max_num_loras,
            max_lora_rank=max_lora_rank,
            max_lora_seq_len=max_lora_seq_len,
            dtype=base.weight.dtype,
            device=base.device,
            has_bias=base.bias is not None,
            quantization_encoding=base.weight.quantization_encoding,
            quant_config=base.quant_config,
            clip_weight=base.clip_weight,
            has_lora_bias=has_lora_bias,
        )

    def lora(self, x: TensorValue) -> TensorValue:
        """Returns the LoRA contribution for ``x``."""
        if (
            self.lora_ids is None
            or self.lora_ranks is None
            or self.lora_grouped_offsets is None
            or self.lora_end_idx is None
        ):
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )
        return sgmv_lora_kernel(
            input=x,
            lora_a=self.lora_A,
            lora_b=self.lora_B,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            grouped_row_offsets=self.lora_grouped_offsets,
            max_lora_seq_len=self.max_lora_seq_len,
            lora_end_idx=self.lora_end_idx,
            bias=self.lora_bias,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        out = super().__call__(x)
        lora_out = self.lora(x)
        assert self.lora_end_idx is not None
        return sliced_add(out, lora_out, self.lora_end_idx)


class StackedLinearLoRA(StackedLinear, LoRAMixin):
    """An unfused QKV :class:`~max.nn.StackedLinear` with a fused LoRA term.

    Adapter weights ``qkv_lora.lora_A`` / ``lora_B_q`` / ``lora_B_kv`` sit at
    ``<attn>.qkv_lora.*`` via the unfused name-omit, matching the keys
    ``LoRAManager`` combines q/k/v adapters into.
    """

    def __init__(
        self,
        in_dim: int,
        out_dims: Sequence[int],
        names: Sequence[str],
        max_num_loras: int,
        max_lora_rank: int,
        max_lora_seq_len: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quant_config: QuantConfig | None = None,
        clip_weight: float | None = None,
    ) -> None:
        if len(out_dims) != 3:
            raise ValueError(
                "StackedLinearLoRA expects three projections (q, k, v); got "
                f"{len(out_dims)}."
            )
        q_dim, k_dim, v_dim = out_dims
        if k_dim != v_dim:
            raise ValueError(
                "K and V projection dims must match for fused QKV LoRA; got "
                f"{k_dim} and {v_dim}."
            )

        super().__init__(
            in_dim=in_dim,
            out_dims=out_dims,
            names=names,
            dtype=dtype,
            device=device,
            stacked=False,
            has_bias=has_bias,
            quant_config=quant_config,
            clip_weight=clip_weight,
        )
        self.max_num_loras = max_num_loras
        self.max_lora_rank = max_lora_rank
        self.max_lora_seq_len = max_lora_seq_len

        # LoRA weights stay in a compute dtype even when the base is quantized.
        lora_dtype = DType.bfloat16 if quant_config is not None else dtype
        self.lora_A = Weight(
            name="qkv_lora.lora_A.weight",
            dtype=lora_dtype,
            shape=[max_num_loras, 3 * max_lora_rank, in_dim],
            device=device,
            _has_alias=True,
        )
        self.lora_B_q = Weight(
            name="qkv_lora.lora_B_q.weight",
            dtype=lora_dtype,
            shape=[max_num_loras, q_dim, max_lora_rank],
            device=device,
            _has_alias=True,
        )
        self.lora_B_kv = Weight(
            name="qkv_lora.lora_B_kv.weight",
            dtype=lora_dtype,
            shape=[2 * max_num_loras, k_dim, max_lora_rank],
            device=device,
            _has_alias=True,
        )

    @classmethod
    def from_base(
        cls,
        base: Module,
        max_num_loras: int,
        max_lora_rank: int,
        max_lora_seq_len: int,
    ) -> StackedLinearLoRA:
        """Builds a :class:`StackedLinearLoRA` mirroring an unfused QKV ``StackedLinear``.

        Reuses the base config so its q/k/v output is unchanged once the
        original child weights load.

        Raises:
            NotImplementedError: If ``base`` is pre-stacked or fp4-quantized.
        """
        assert isinstance(base, StackedLinear)
        if base._stacked:
            raise NotImplementedError(
                "LoRA is not supported for pre-stacked QKV "
                "projections; only the unfused q/k/v form is supported."
            )
        if base._quant_config is not None and base._quant_config.is_fp4:
            raise NotImplementedError(
                "LoRA is not yet supported for fp4-quantized projections."
            )
        first = base._child(base._names[0])
        return cls(
            in_dim=base._in_dim,
            out_dims=base._out_dims,
            names=base._names,
            max_num_loras=max_num_loras,
            max_lora_rank=max_lora_rank,
            max_lora_seq_len=max_lora_seq_len,
            dtype=first.weight.dtype,
            device=first.device,
            has_bias=base._has_bias,
            quant_config=base._quant_config,
            clip_weight=base._clip_weight,
        )

    def qkv_lora(self, x: TensorValue) -> TensorValue:
        """Returns the fused ``[q|k|v]`` LoRA contribution for ``x``."""
        if (
            self.lora_ids is None
            or self.lora_ranks is None
            or self.lora_grouped_offsets is None
            or self.lora_end_idx is None
            or self.lora_ids_kv is None
            or self.lora_grouped_offsets_kv is None
        ):
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )
        return sgmv_qkv_lora_fused(
            input=x,
            lora_a=self.lora_A,
            lora_b_q=self.lora_B_q,
            lora_b_kv=self.lora_B_kv,
            lora_ids=self.lora_ids,
            lora_ranks=self.lora_ranks,
            lora_grouped_offsets=self.lora_grouped_offsets,
            lora_end_idx=self.lora_end_idx,
            lora_ids_kv=self.lora_ids_kv,
            lora_grouped_offsets_kv=self.lora_grouped_offsets_kv,
            max_lora_seq_len=self.max_lora_seq_len,
            max_rank=self.max_lora_rank,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        out = super().__call__(x)
        lora_out = self.qkv_lora(x)
        assert self.lora_end_idx is not None
        return sliced_add(out, lora_out, self.lora_end_idx)
