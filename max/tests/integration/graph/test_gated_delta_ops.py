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
"""Python-level integration tests for the Gated DeltaNet (GDN) GPU kernels.

Tests call through the @compiler.register path (MOGGKernelAPI) via ops.custom()
and verify output shapes and basic numerical properties.

GDN ops are registered in MOGGKernelAPI and are part of the built-in engine
kernel library — no custom_extensions are needed.

Kernel op signatures
--------------------
gated_delta_conv1d_fwd (no parameters):
  inputs:  qkv_input_ragged [T, C], conv_weight [C, K],
           conv_state_in [B, C, K-1], input_row_offsets [B+1] uint32
  outputs: conv_output_ragged [T, C], conv_state_out [B, C, K-1]
  note:    only K=4 is compiled in the current build

gated_delta_recurrence_fwd (parameter delta_softplus: Bool = False):
  inputs:  qkv_conv_output [T, D], decay_per_token [T, H],
           beta_per_token [T, H], recurrent_state_in [B, H, KD, VD],
           input_row_offsets [B+1] uint32
  outputs: recurrence_output [T, H*VD], recurrent_state_out [B, H, KD, VD]
  note:    only (KD, VD) = (128, 128) is compiled in the current build
           D = H*VD + 2*(num_key_heads*KD)
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# ── Conv1d test dimensions ────────────────────────────────────────────────────
_CONV1D_BATCH = 2
_CONV1D_TOTAL_SEQ_LEN = 8
_CONV1D_CONV_DIM = 128  # multiple of CONV1D_BLOCK_DIM (128)
_CONV1D_KERNEL_SIZE = 4  # only supported size in current build
# Two equal-length sequences: [4, 4] → exclusive prefix offsets [0, 4, 8]
_CONV1D_OFFSETS = np.array([0, 4, 8], dtype=np.uint32)

# ── Recurrence test dimensions ────────────────────────────────────────────────
_RECUR_BATCH = 2
_RECUR_TOTAL_SEQ_LEN = 8
_RECUR_NUM_VALUE_HEADS = 2
_RECUR_KEY_HEAD_DIM = 128  # only (128,128) compiled in current build
_RECUR_VALUE_HEAD_DIM = 128
_RECUR_NUM_KEY_HEADS = 2
# conv_dim = H*VD + 2*(num_key_heads*KD)
_RECUR_VALUE_DIM = _RECUR_NUM_VALUE_HEADS * _RECUR_VALUE_HEAD_DIM  # 256
_RECUR_KEY_DIM = _RECUR_NUM_KEY_HEADS * _RECUR_KEY_HEAD_DIM  # 256
_RECUR_CONV_DIM = _RECUR_VALUE_DIM + 2 * _RECUR_KEY_DIM  # 768
# Two equal-length sequences: [4, 4] → exclusive prefix offsets [0, 4, 8]
_RECUR_OFFSETS = np.array([0, 4, 8], dtype=np.uint32)


def test_gated_delta_conv1d_fwd_shapes(
    session: InferenceSession,
) -> None:
    """gated_delta_conv1d_fwd: output shapes and finiteness via the Graph API."""
    gpu = DeviceRef.GPU()
    T = _CONV1D_TOTAL_SEQ_LEN
    C = _CONV1D_CONV_DIM
    K = _CONV1D_KERNEL_SIZE
    B = _CONV1D_BATCH

    with Graph(
        "gdn_conv1d_shape_test",
        input_types=[
            TensorType(DType.float32, [T, C], device=gpu),  # qkv_input_ragged
            TensorType(DType.float32, [C, K], device=gpu),  # conv_weight
            TensorType(
                DType.float32, [B, C, K - 1], device=gpu
            ),  # conv_state_in
            TensorType(DType.uint32, [B + 1], device=gpu),  # input_row_offsets
        ],
    ) as graph:
        qkv_in, conv_w, state_in, offsets = [inp.tensor for inp in graph.inputs]
        results = ops.custom(
            "gated_delta_conv1d_fwd",
            device=gpu,
            values=[qkv_in, conv_w, state_in, offsets],
            out_types=[
                TensorType(DType.float32, [T, C], device=gpu),
                TensorType(DType.float32, [B, C, K - 1], device=gpu),
            ],
        )
        graph.output(results[0], results[1])

    model = session.load(graph)

    rng = np.random.default_rng(42)
    gpu_device = model.input_devices[0]
    outputs = model.execute(
        md.Buffer.from_numpy(rng.standard_normal((T, C)).astype(np.float32)).to(
            gpu_device
        ),
        md.Buffer.from_numpy(rng.standard_normal((C, K)).astype(np.float32)).to(
            gpu_device
        ),
        md.Buffer.from_numpy(np.zeros((B, C, K - 1), dtype=np.float32)).to(
            gpu_device
        ),
        md.Buffer.from_numpy(_CONV1D_OFFSETS).to(gpu_device),
    )

    conv_out = torch.from_dlpack(outputs[0])
    state_out = torch.from_dlpack(outputs[1])

    assert conv_out.shape == torch.Size([T, C]), (
        f"conv_output_ragged: expected ({T}, {C}), got {tuple(conv_out.shape)}"
    )
    assert state_out.shape == torch.Size([B, C, K - 1]), (
        f"conv_state_out: expected ({B}, {C}, {K - 1}),"
        f" got {tuple(state_out.shape)}"
    )
    assert torch.all(torch.isfinite(conv_out)), (
        "conv_output_ragged contains non-finite values"
    )
    assert torch.all(torch.isfinite(state_out)), (
        "conv_state_out contains non-finite values"
    )


def test_gated_delta_recurrence_fwd_shapes(
    session: InferenceSession,
) -> None:
    """gated_delta_recurrence_fwd: output shapes and finiteness via Graph API."""
    gpu = DeviceRef.GPU()
    T = _RECUR_TOTAL_SEQ_LEN
    B = _RECUR_BATCH
    H = _RECUR_NUM_VALUE_HEADS
    KD = _RECUR_KEY_HEAD_DIM
    VD = _RECUR_VALUE_HEAD_DIM
    D = _RECUR_CONV_DIM

    with Graph(
        "gdn_recurrence_shape_test",
        input_types=[
            TensorType(DType.float32, [T, D], device=gpu),  # qkv_conv_output
            TensorType(DType.float32, [T, H], device=gpu),  # decay_per_token
            TensorType(DType.float32, [T, H], device=gpu),  # beta_per_token
            TensorType(
                DType.float32, [B, H, KD, VD], device=gpu
            ),  # recurrent_state_in
            TensorType(DType.uint32, [B + 1], device=gpu),  # input_row_offsets
        ],
    ) as graph:
        qkv, decay, beta, r_state, offsets = [
            inp.tensor for inp in graph.inputs
        ]
        results = ops.custom(
            "gated_delta_recurrence_fwd",
            device=gpu,
            values=[qkv, decay, beta, r_state, offsets],
            out_types=[
                TensorType(DType.float32, [T, H * VD], device=gpu),
                TensorType(DType.float32, [B, H, KD, VD], device=gpu),
            ],
        )
        graph.output(results[0], results[1])

    model = session.load(graph)

    rng = np.random.default_rng(42)
    # decay and beta should be in (0, 1) for numerical stability
    decay_np = (
        np.abs(rng.standard_normal((T, H)).astype(np.float32)) * 0.5
    ).clip(0.0, 1.0)
    beta_np = (
        np.abs(rng.standard_normal((T, H)).astype(np.float32)) * 0.5
    ).clip(0.0, 1.0)

    gpu_device = model.input_devices[0]
    outputs = model.execute(
        md.Buffer.from_numpy(rng.standard_normal((T, D)).astype(np.float32)).to(
            gpu_device
        ),
        md.Buffer.from_numpy(decay_np).to(gpu_device),
        md.Buffer.from_numpy(beta_np).to(gpu_device),
        md.Buffer.from_numpy(np.zeros((B, H, KD, VD), dtype=np.float32)).to(
            gpu_device
        ),
        md.Buffer.from_numpy(_RECUR_OFFSETS).to(gpu_device),
    )

    rec_out = torch.from_dlpack(outputs[0])
    state_out = torch.from_dlpack(outputs[1])

    assert rec_out.shape == torch.Size([T, H * VD]), (
        f"recurrence_output: expected ({T}, {H * VD}),"
        f" got {tuple(rec_out.shape)}"
    )
    assert state_out.shape == torch.Size([B, H, KD, VD]), (
        f"recurrent_state_out: expected ({B}, {H}, {KD}, {VD}),"
        f" got {tuple(state_out.shape)}"
    )
    assert torch.all(torch.isfinite(rec_out)), (
        "recurrence_output contains non-finite values"
    )
    assert torch.all(torch.isfinite(state_out)), (
        "recurrent_state_out contains non-finite values"
    )
