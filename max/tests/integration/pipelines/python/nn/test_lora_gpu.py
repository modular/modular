# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Tests for LoRA layers that compare MAX vs PyTorch outputs.
"""

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from max.driver import CPU, Accelerator, Device, DLPackArray, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
from max.graph.weights.weights import _cast_to_dtype
from max.nn import AttentionWithRopeAndLoRA, LinearLoRA, RotaryEmbedding
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from test_common.context_utils import create_text_context

DTYPE = DType.bfloat16
TORCH_DTYPE = torch.bfloat16
ACCURACY_RTOL = 15e-3
ACCURACY_ATOL = 3e-1


def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely convert PyTorch tensor to NumPy array, handling bfloat16."""
    if tensor.dtype == torch.bfloat16:
        # Convert bfloat16 to float32 for NumPy compatibility
        return tensor.to(torch.float32).cpu().numpy()
    else:
        return tensor.cpu().numpy()


def generate_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int = 1234,
    scale: float = 0.01,
) -> torch.Tensor:
    """Generate reproducible random tensor."""
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype) * scale


def to_max_weight(
    weight: torch.Tensor, name: str, device: Device
) -> WeightData:
    return WeightData(
        data=Tensor.from_numpy(safe_tensor_to_numpy(weight)).to(device),
        shape=Shape(weight.shape),
        name=name,
        dtype=DType.from_torch(weight.dtype),
    )


def create_lora_buffers(
    weight_shape,  # noqa: ANN001
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    device: Device,
    alpha: int,
    max_adapters: int = 2,
) -> tuple[Tensor, Tensor]:
    """Create LoRA adapter buffers."""
    rank = lora_A.shape[0]
    lora_A_buffer = Tensor.zeros(
        (max_adapters, rank, weight_shape[1]),
        dtype=DType.bfloat16,
        device=device,
    )
    lora_B_buffer = Tensor.zeros(
        (max_adapters, weight_shape[0], rank),
        dtype=DType.bfloat16,
        device=device,
    )
    lora_B = lora_B * alpha / rank
    lora_A_buffer[0, :, :].inplace_copy_from(
        _cast_to_dtype(
            Tensor.from_numpy(safe_tensor_to_numpy(lora_A)),
            DType.float32,
            DType.bfloat16,
        ).to(device)
    )
    lora_B_buffer[0, :, :].inplace_copy_from(
        _cast_to_dtype(
            Tensor.from_numpy(safe_tensor_to_numpy(lora_B)),
            DType.float32,
            DType.bfloat16,
        ).to(device)
    )
    return lora_A_buffer, lora_B_buffer


def setup_session_and_device(is_gpu: bool):
    """Setup inference session and device."""
    session = (
        InferenceSession(devices=[Accelerator(0)])
        if is_gpu
        else InferenceSession()
    )
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    device = Accelerator(0) if is_gpu else CPU()
    return session, device_ref, device


@dataclasses.dataclass
class AttentionTestConfig:
    """Configuration for attention LoRA tests."""

    hidden_size: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    rank: int
    alpha: int
    seq_len: int = 8
    theta: float = 10000.0


@dataclasses.dataclass
class AttentionWeights:
    """Container for attention weights and LoRA adapters."""

    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    o_weight: torch.Tensor
    q_lora_A: torch.Tensor
    q_lora_B: torch.Tensor
    k_lora_A: torch.Tensor
    k_lora_B: torch.Tensor
    v_lora_A: torch.Tensor
    v_lora_B: torch.Tensor
    o_lora_A: torch.Tensor
    o_lora_B: torch.Tensor


def create_attention_weights(config: AttentionTestConfig) -> AttentionWeights:
    """Generate attention weights for testing."""
    return AttentionWeights(
        q_weight=generate_tensor(
            (config.n_q_heads * config.head_dim, config.hidden_size),
            TORCH_DTYPE,
            seed=42,
        ),
        k_weight=generate_tensor(
            (config.n_kv_heads * config.head_dim, config.hidden_size),
            TORCH_DTYPE,
            seed=43,
        ),
        v_weight=generate_tensor(
            (config.n_kv_heads * config.head_dim, config.hidden_size),
            TORCH_DTYPE,
            seed=44,
        ),
        o_weight=generate_tensor(
            (config.hidden_size, config.n_q_heads * config.head_dim),
            TORCH_DTYPE,
            seed=45,
        ),
        q_lora_A=generate_tensor(
            (config.rank, config.hidden_size), TORCH_DTYPE, seed=46
        ),
        q_lora_B=generate_tensor(
            (config.n_q_heads * config.head_dim, config.rank),
            TORCH_DTYPE,
            seed=47,
        ),
        k_lora_A=generate_tensor(
            (config.rank, config.hidden_size), TORCH_DTYPE, seed=48
        ),
        k_lora_B=generate_tensor(
            (config.n_kv_heads * config.head_dim, config.rank),
            TORCH_DTYPE,
            seed=49,
        ),
        v_lora_A=generate_tensor(
            (config.rank, config.hidden_size), TORCH_DTYPE, seed=50
        ),
        v_lora_B=generate_tensor(
            (config.n_kv_heads * config.head_dim, config.rank),
            TORCH_DTYPE,
            seed=51,
        ),
        o_lora_A=generate_tensor(
            (config.rank, config.n_q_heads * config.head_dim),
            TORCH_DTYPE,
            seed=52,
        ),
        o_lora_B=generate_tensor(
            (config.hidden_size, config.rank), TORCH_DTYPE, seed=53
        ),
    )


class TorchLinearLoRA(nn.Module):
    """PyTorch reference implementation of Linear LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
    ):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        return base_output + lora_output * self.scaling


class TorchRoPEAttentionWithLoRA(nn.Module):
    """PyTorch reference RoPE attention with LoRA"""

    def __init__(
        self,
        hidden_size: int,
        n_q_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        theta: float = 10000.0,
        max_seq_len: int = 512,
        rank: int = 8,
        alpha: int = 16,
    ):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads or n_q_heads
        self.head_dim = head_dim or hidden_size // n_q_heads
        self.hidden_size = hidden_size
        self.theta = theta
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(
            hidden_size, self.n_q_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_q_heads * self.head_dim, hidden_size, bias=False
        )

        # LoRA adapters for each projection
        self.q_lora_A = nn.Parameter(torch.randn(rank, hidden_size))
        self.q_lora_B = nn.Parameter(
            torch.randn(self.n_q_heads * self.head_dim, rank)
        )
        self.k_lora_A = nn.Parameter(torch.randn(rank, hidden_size))
        self.k_lora_B = nn.Parameter(
            torch.randn(self.n_kv_heads * self.head_dim, rank)
        )
        self.v_lora_A = nn.Parameter(torch.randn(rank, hidden_size))
        self.v_lora_B = nn.Parameter(
            torch.randn(self.n_kv_heads * self.head_dim, rank)
        )
        self.o_lora_A = nn.Parameter(
            torch.randn(rank, self.n_q_heads * self.head_dim)
        )
        self.o_lora_B = nn.Parameter(torch.randn(hidden_size, rank))

        self.lora_scaling = alpha / rank

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self._cos_sin_cached = None
        self._seq_len_cached = 0

    def _get_cos_sin(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(
                t, self.inv_freq.to(device)
            )  # [seq_len, head_dim//2]

            # Create interleaved cos/sin like MAX implementation
            cos_freqs = freqs.cos()  # [seq_len, head_dim//2]
            sin_freqs = freqs.sin()  # [seq_len, head_dim//2]

            # Interleave cos and sin: [cos0, sin0, cos1, sin1, ...]
            cos_sin_interleaved = torch.stack(
                [cos_freqs, sin_freqs], dim=-1
            )  # [seq_len, head_dim//2, 2]
            self._cos_sin_cached = cos_sin_interleaved.reshape(
                seq_len, -1
            )  # [seq_len, head_dim]

        assert self._cos_sin_cached is not None
        cos_sin = self._cos_sin_cached[:seq_len].to(device)
        return cos_sin

    def _apply_rotary_pos_emb(self, x, cos_sin):  # noqa: ANN001
        cos_sin = cos_sin.unsqueeze(0).unsqueeze(0)  # Add batch and head dims

        # Extract interleaved cos and sin values
        # cos_sin format: [cos0, sin0, cos1, sin1, ...]
        cos = cos_sin[..., ::2]  # Extract cos values: [cos0, cos1, cos2, ...]
        sin = cos_sin[..., 1::2]  # Extract sin values: [sin0, sin1, sin2, ...]

        # Split x into even/odd components for rotation
        x_real = x[..., ::2]  # Real part: [x0, x2, x4, ...]
        x_imag = x[..., 1::2]  # Imag part: [x1, x3, x5, ...]

        # Apply rotation: new_real = x_real * cos - x_imag * sin, new_imag = x_real * sin + x_imag * cos
        rotated_real = x_real * cos - x_imag * sin
        rotated_imag = x_real * sin + x_imag * cos

        # Interleave the results to match MAX's output format: [real0, imag0, real1, imag1, ...]
        result = torch.stack([rotated_real, rotated_imag], dim=-1)
        return result.reshape(x.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        q_proj_out = self.q_proj(x)
        q = (
            q_proj_out
            + (x @ self.q_lora_A.T) @ self.q_lora_B.T * self.lora_scaling
        )
        q = (
            q_proj_out
            + (x @ self.q_lora_A.T) @ self.q_lora_B.T * self.lora_scaling
        )
        k = (
            self.k_proj(x)
            + (x @ self.k_lora_A.T) @ self.k_lora_B.T * self.lora_scaling
        )
        v = (
            self.v_proj(x)
            + (x @ self.v_lora_A.T) @ self.v_lora_B.T * self.lora_scaling
        )

        q = q.view(
            batch_size, seq_len, self.n_q_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos_sin = self._get_cos_sin(seq_len, x.device)
        q = self._apply_rotary_pos_emb(q, cos_sin)
        k = self._apply_rotary_pos_emb(k, cos_sin)

        if self.n_q_heads != self.n_kv_heads:
            repeat_factor = self.n_q_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply causal mask to match MAX's MHAMaskVariant.CAUSAL_MASK
        seq_len = attn_weights.size(-1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_weights.device), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.bool(), float("-inf")
        )

        attn_weights = torch.softmax(attn_weights, dim=-1).to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        output = (
            self.o_proj(attn_output)
            + (attn_output @ self.o_lora_A.T)
            @ self.o_lora_B.T
            * self.lora_scaling
        )

        return output


def linear_lora_max_output(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    x: torch.Tensor,
    rank: int,
    alpha: int,
    is_gpu: bool = False,
) -> list[Tensor]:
    """Generate MAX LinearLoRA output with full LoRA functionality."""
    max_adapters = 2
    session, device_ref, device = setup_session_and_device(is_gpu)

    lora_A_buf, lora_B_buf = create_lora_buffers(
        base_weight.shape, lora_A, lora_B, device, alpha, max_adapters
    )

    state_dict: Mapping[str, Union[DLPackArray, WeightData]] = {
        "weight": to_max_weight(base_weight, "weight", CPU()),
        "lora_A.weight": lora_A_buf,
        "lora_B.weight": lora_B_buf,
    }

    max_lora = LinearLoRA(
        in_dim=base_weight.shape[1],
        out_dim=base_weight.shape[0],
        max_lora_rank=rank,
        max_num_loras=max_adapters,
        dtype=DTYPE,
        device=device_ref,
    )
    max_lora.load_state_dict(state_dict)

    with Graph(
        "LinearLoRA",
        input_types=[
            TensorType(DTYPE, x.shape, device=device_ref),
            TensorType(DType.int32, ["lora_ids"], device=device_ref),
            TensorType(DType.uint32, ["lora_ranks"], device=device_ref),
            TensorType(DType.uint32, ["input_row_offsets"], device=device_ref),
        ],
    ) as graph:
        x_input, lora_ids_input, lora_ranks_input, input_row_offsets_input = (
            graph.inputs
        )
        max_lora.set_lora_batch_info(
            lora_ids_input.tensor, lora_ranks_input.tensor
        )
        output = max_lora.apply_lora(
            x_input.tensor, input_row_offsets_input.tensor
        ).cast(DType.float32)
        graph.output(output)

    compiled = session.load(graph, weights_registry=max_lora.state_dict())

    batch_size = x.shape[0] * x.shape[1]
    lora_ids = np.zeros(batch_size, dtype=np.int32)
    lora_ranks = np.full(batch_size, rank, dtype=np.uint32)
    input_row_offsets = np.arange(batch_size + 1, dtype=np.uint32)

    x_tensor = (
        _cast_to_dtype(
            Tensor.from_numpy(safe_tensor_to_numpy(x)),
            DType.float32,
            DType.bfloat16,
        ).to(device)
        if is_gpu
        else x
    )

    return compiled.execute(
        x_tensor,
        Tensor.from_numpy(lora_ids).to(device),
        Tensor.from_numpy(lora_ranks).to(device),
        Tensor.from_numpy(input_row_offsets).to(device),
    )


def compare_linear_lora_outputs(
    in_dim: int,
    out_dim: int,
    rank: int,
    alpha: int,
    is_gpu: bool = False,
) -> None:
    """Compare MAX LinearLoRA full vs PyTorch LinearLoRA outputs."""

    base_weight = generate_tensor((out_dim, in_dim), TORCH_DTYPE, seed=42)
    lora_A = generate_tensor((rank, in_dim), TORCH_DTYPE, seed=43)
    lora_B = generate_tensor((out_dim, rank), TORCH_DTYPE, seed=44)
    x = generate_tensor((2, in_dim), TORCH_DTYPE, seed=45)

    max_output = linear_lora_max_output(
        base_weight, lora_A, lora_B, x, rank, alpha, is_gpu=is_gpu
    )

    device = "cuda" if is_gpu else "cpu"
    torch_lora = TorchLinearLoRA(in_dim, out_dim, rank, alpha)
    torch_lora.base_layer.weight = nn.Parameter(base_weight.to(device))
    torch_lora.lora_A = nn.Parameter(lora_A.to(device))
    torch_lora.lora_B = nn.Parameter(lora_B.to(device))

    torch_lora_output = torch_lora(x.to(device)).detach().cpu()

    torch.testing.assert_close(
        torch_lora_output,
        torch.tensor(max_output[0].to_numpy()).to(TORCH_DTYPE),
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def attention_lora_max_output(
    weights: AttentionWeights,
    x: torch.Tensor,
    config: AttentionTestConfig,
    is_gpu: bool = False,
) -> torch.Tensor:
    """Generate MAX AttentionWithRopeAndLoRA output with LoRA functionality."""
    session, device_ref, device = setup_session_and_device(is_gpu)

    hidden_size = x.shape[-1]
    seq_len = x.shape[1]
    max_adapters = 2

    # Create LoRA buffers for all projections
    q_lora_A_buf, q_lora_B_buf = create_lora_buffers(
        weights.q_weight.shape,
        weights.q_lora_A,
        weights.q_lora_B,
        device,
        config.alpha,
        max_adapters,
    )
    k_lora_A_buf, k_lora_B_buf = create_lora_buffers(
        weights.k_weight.shape,
        weights.k_lora_A,
        weights.k_lora_B,
        device,
        config.alpha,
        max_adapters,
    )
    v_lora_A_buf, v_lora_B_buf = create_lora_buffers(
        weights.v_weight.shape,
        weights.v_lora_A,
        weights.v_lora_B,
        device,
        config.alpha,
        max_adapters,
    )
    o_lora_A_buf, o_lora_B_buf = create_lora_buffers(
        weights.o_weight.shape,
        weights.o_lora_A,
        weights.o_lora_B,
        device,
        config.alpha,
        max_adapters,
    )

    state_dict: Mapping[str, Union[DLPackArray, WeightData]] = {
        "q_proj.weight": to_max_weight(
            weights.q_weight, "q_proj.weight", CPU()
        ),
        "k_proj.weight": to_max_weight(
            weights.k_weight, "k_proj.weight", CPU()
        ),
        "v_proj.weight": to_max_weight(
            weights.v_weight, "v_proj.weight", CPU()
        ),
        "o_proj.weight": to_max_weight(
            weights.o_weight, "o_proj.weight", CPU()
        ),
        "q_proj.lora_A.weight": q_lora_A_buf,
        "q_proj.lora_B.weight": q_lora_B_buf,
        "k_proj.lora_A.weight": k_lora_A_buf,
        "k_proj.lora_B.weight": k_lora_B_buf,
        "v_proj.lora_A.weight": v_lora_A_buf,
        "v_proj.lora_B.weight": v_lora_B_buf,
        "o_proj.lora_A.weight": o_lora_A_buf,
        "o_proj.lora_B.weight": o_lora_B_buf,
    }

    rope = RotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.n_q_heads,
        theta=config.theta,
        max_seq_len=seq_len * 2,
        device=device_ref,
    )

    kv_params = KVCacheParams(
        dtype=DTYPE,
        page_size=128,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        cache_strategy=KVCacheStrategy.PAGED,
    )

    linear_lora_cls = functools.partial(
        LinearLoRA,
        max_num_loras=max_adapters,
        max_lora_rank=config.rank,
    )

    max_attention = AttentionWithRopeAndLoRA(
        rope=rope,
        num_attention_heads=config.n_q_heads,
        num_key_value_heads=config.n_kv_heads,
        hidden_size=hidden_size,
        kv_params=kv_params,
        devices=[device_ref],
        dtype=DTYPE,
        linear_cls=linear_lora_cls,
        has_bias=False,
    )
    max_attention.load_state_dict(state_dict)

    device = Accelerator(0) if is_gpu else CPU()

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        cache_memory=1024 * 1024,
        max_batch_size=x.shape[0],
        max_seq_len=seq_len * 2,
        num_layers=1,
        devices=[device],
        session=session,
    )

    blocks_type, cache_lengths_type, lookup_table_type, max_lengths_type = (
        kv_manager.input_symbols()[0]
    )

    with Graph(
        "AttentionLoRA",
        input_types=[
            TensorType(
                DTYPE, [x.shape[0] * x.shape[1], hidden_size], device=device_ref
            ),
            TensorType(DType.uint32, [x.shape[0] + 1], device=device_ref),
            TensorType(DType.int32, ["lora_ids"], device=device_ref),
            TensorType(DType.uint32, ["lora_ranks"], device=device_ref),
            TensorType(
                DType.uint32, ["lora_input_row_offsets"], device=device_ref
            ),
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_lengths_type,
        ],
    ) as graph:
        (
            x_input,
            input_row_offsets,
            lora_ids_input,
            lora_ranks_input,
            lora_input_row_offsets_input,
            blocks,
            cache_lengths,
            lookup_table,
            max_lengths,
        ) = graph.inputs

        for proj in [
            max_attention.q_proj,
            max_attention.k_proj,
            max_attention.v_proj,
            max_attention.o_proj,
        ]:
            if hasattr(proj, "set_lora_batch_info"):
                proj.set_lora_batch_info(
                    lora_ids_input.tensor, lora_ranks_input.tensor
                )

        fetch_op = FetchPagedKVCacheCollection(kv_params)
        kv_collection = fetch_op(
            blocks.tensor,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_lengths.tensor,
        )

        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        full_attention_output = max_attention(
            layer_idx,
            x_input.tensor,
            kv_collection,
            freqs_cis=rope.freqs_cis,
            input_row_offsets=input_row_offsets.tensor,
        ).cast(DType.float32)

        graph.output(full_attention_output)

    compiled = session.load(graph, weights_registry=max_attention.state_dict())

    batch_size = x.shape[0]
    x_flattened = x.reshape(-1, hidden_size)

    attention_input_row_offsets = np.array(
        [0, seq_len, seq_len * 2], dtype=np.uint32
    )

    lora_ids = np.zeros(batch_size, dtype=np.int32)
    lora_ranks = np.full(batch_size, config.rank, dtype=np.uint32)
    lora_input_row_offsets = np.array(
        [0, seq_len, seq_len * 2], dtype=np.uint32
    )

    batch = [create_text_context(np.empty(seq_len)) for _ in range(batch_size)]

    for context in batch:
        kv_manager.external_claim(context.request_id)

    blocks, cache_lengths, lookup_table_tensor, max_lengths_buf = (
        kv_manager.fetch(batch)[0]  # type: ignore
    )

    x_tensor = (
        _cast_to_dtype(
            Tensor.from_numpy(safe_tensor_to_numpy(x_flattened)),
            DType.float32,
            DType.bfloat16,
        ).to(device)
        if is_gpu
        else x_flattened
    )

    return compiled.execute(
        x_tensor,
        Tensor.from_numpy(attention_input_row_offsets).to(device),
        Tensor.from_numpy(lora_ids).to(device),
        Tensor.from_numpy(lora_ranks).to(device),
        Tensor.from_numpy(lora_input_row_offsets).to(device),
        blocks,
        cache_lengths,
        lookup_table_tensor,
        max_lengths_buf,
    )


def compare_attention_lora_outputs(
    config: AttentionTestConfig, is_gpu: bool = False
) -> None:
    """Compare MAX AttentionWithRopeAndLoRA vs PyTorch RoPE attention with LoRA."""
    weights = create_attention_weights(config)
    x = generate_tensor(
        (2, config.seq_len, config.hidden_size), TORCH_DTYPE, seed=54
    )

    max_output = attention_lora_max_output(weights, x, config, is_gpu=is_gpu)

    device = "cuda" if is_gpu else "cpu"
    torch_attention = TorchRoPEAttentionWithLoRA(
        config.hidden_size,
        config.n_q_heads,
        config.n_kv_heads,
        config.head_dim,
        config.theta,
        config.seq_len * 2,
        config.rank,
        config.alpha,
    )

    # Set weights
    torch_attention.q_proj.weight = nn.Parameter(weights.q_weight.to(device))
    torch_attention.k_proj.weight = nn.Parameter(weights.k_weight.to(device))
    torch_attention.v_proj.weight = nn.Parameter(weights.v_weight.to(device))
    torch_attention.o_proj.weight = nn.Parameter(weights.o_weight.to(device))
    torch_attention.q_lora_A = nn.Parameter(weights.q_lora_A.to(device))
    torch_attention.q_lora_B = nn.Parameter(weights.q_lora_B.to(device))
    torch_attention.k_lora_A = nn.Parameter(weights.k_lora_A.to(device))
    torch_attention.k_lora_B = nn.Parameter(weights.k_lora_B.to(device))
    torch_attention.v_lora_A = nn.Parameter(weights.v_lora_A.to(device))
    torch_attention.v_lora_B = nn.Parameter(weights.v_lora_B.to(device))
    torch_attention.o_lora_A = nn.Parameter(weights.o_lora_A.to(device))
    torch_attention.o_lora_B = nn.Parameter(weights.o_lora_B.to(device))

    torch_output = torch_attention(x.to(device)).detach().cpu()
    torch_output_numpy = safe_tensor_to_numpy(torch_output)

    max_output_reshaped = max_output[0].to_numpy()
    max_output_reshaped = max_output_reshaped.reshape(
        (x.shape[0], x.shape[1], config.hidden_size)
    )

    torch.testing.assert_close(
        torch_output_numpy,
        max_output_reshaped,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def test_attention_lora() -> None:
    is_gpu = True
    config1 = AttentionTestConfig(
        hidden_size=128 * 4,
        n_q_heads=4,
        n_kv_heads=4,
        head_dim=128,
        rank=8,
        alpha=8,  # usually this is 16, but bfloat16 precision gets weird with 16
    )
    compare_attention_lora_outputs(config1, is_gpu=is_gpu)

    config2 = AttentionTestConfig(
        hidden_size=128 * 8,
        n_q_heads=8,
        n_kv_heads=2,
        head_dim=128,
        rank=8,
        alpha=8,  # usually this is 16, but bfloat16 precision gets weird with 16
    )
    compare_attention_lora_outputs(config2, is_gpu=is_gpu)


def test_linear_lora() -> None:
    is_gpu = True
    compare_linear_lora_outputs(512, 1024, 8, 16, is_gpu=is_gpu)
    compare_linear_lora_outputs(1024, 512, 8, 16, is_gpu=is_gpu)
