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
"""Python wrappers for the Gated DeltaNet two-pass prefill kernels.

Provides two graph-level wrappers that call the Mojo ops registered in
state_space.mojopkg:

  gated_delta_conv1d_fwd()
    Pass 1: causal depthwise conv1d over a ragged batch of sequences.
    Inputs use seqlen-first [total_seq_len, conv_dim] layout.

  gated_delta_recurrence_fwd()
    Pass 2: gated delta rule recurrence over the conv1d outputs.
    Produces [total_seq_len, value_dim] output and updated recurrent state.

get_state_space_paths() locates the state_space.mojopkg via
MODULAR_MOJO_MAX_IMPORT_PATH.  It is memoised and called once at Graph
construction time to pre-register the extension before any op is verified.

Usage
-----
    from .functional_ops import (
        gated_delta_conv1d_fwd,
        gated_delta_recurrence_fwd,
        get_state_space_paths,
    )

    # Pass 1
    conv_output, new_conv_state = gated_delta_conv1d_fwd(
        qkv_input_ragged=qkv_f32,          # [total_N, conv_dim]
        conv_weight=conv_weight_flat,       # [conv_dim, K]
        conv_state_in=conv_state,           # [B, conv_dim, K-1]
        input_row_offsets=offsets_uint32,   # [B+1]
    )

    # Pass 2
    recurrence_output, new_recurrent_state = gated_delta_recurrence_fwd(
        qkv_conv_output=conv_output,        # [total_N, conv_dim]
        decay_per_token=decay,              # [total_N, nv]
        beta_per_token=beta,                # [total_N, nv]
        recurrent_state_in=recurrent_state, # [B, nv, kd, vd]
        input_row_offsets=offsets_uint32,   # [B+1]
    )
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import cast

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops

logger = logging.getLogger("max.pipelines.qwen3_5")

_MODULAR_MOJO_MAX_IMPORT_PATH_ENV_VAR = "MODULAR_MOJO_MAX_IMPORT_PATH"


# TODO: Remove (along with logger, env var constant, and unused imports) once kernels are in main.
@functools.cache
def get_state_space_paths() -> tuple[Path, ...]:
    """Locate state_space.mojopkg files via MODULAR_MOJO_MAX_IMPORT_PATH.

    Searches comma-separated directory or .mojopkg entries from the env var
    and returns all resolved paths whose filename contains "state_space".

    **IMPORTANT**: This function caches its result for the process lifetime.
    MODULAR_MOJO_MAX_IMPORT_PATH must be set BEFORE this function is called.
    If the environment variable is modified after the first call (common in
    test harnesses), the cache will be stale and tests may silently run the
    wrong code path. Call get_state_space_paths.cache_clear() between tests
    if needed.

    Returns:
        Tuple of resolved paths to state_space.mojopkg files, or an empty
        tuple if the environment variable is not set or no matching files
        are found.  An empty tuple means the kernel path will not be used
        (the while_loop fallback is selected instead).
    """
    import_path_env = os.environ.get(_MODULAR_MOJO_MAX_IMPORT_PATH_ENV_VAR, "")
    if not import_path_env:
        logger.warning(
            "MODULAR_MOJO_MAX_IMPORT_PATH is not set; gated_delta kernels"
            " will not be available for the prefill path."
        )
        return ()

    found_paths: list[Path] = []
    for raw_entry in import_path_env.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            resolved = Path.cwd() / entry_path
            entry_path = resolved if resolved.exists() else entry_path
        if not entry_path.exists():
            continue
        if entry_path.suffix == ".mojopkg" and "state_space" in entry_path.name:
            found_paths.append(entry_path.resolve())
        elif entry_path.is_dir():
            for mojopkg_path in entry_path.rglob("*.mojopkg"):
                if "state_space" in mojopkg_path.name and (
                    mojopkg_path.is_file() or mojopkg_path.is_symlink()
                ):
                    found_paths.append(mojopkg_path.resolve())

    logger.debug(
        "gated_delta kernels: found %d state_space.mojopkg path(s)",
        len(found_paths),
    )
    return tuple(found_paths)


def gated_delta_conv1d_fwd(
    qkv_input_ragged: TensorValue,
    conv_weight: TensorValue,
    conv_state_in: TensorValue,
    input_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Pass 1: causal depthwise conv1d over a ragged batch.

    Computes the causal sliding-window convolution for all tokens in all
    sequences simultaneously (one GPU thread per channel per batch item).
    Updates the per-sequence conv state.

    Tensor shapes (seqlen-first layout):
        qkv_input_ragged  : [total_seq_len, conv_dim]             float32
        conv_weight       : [conv_dim, kernel_size]               float32
        conv_state_in     : [batch_size, conv_dim, kernel_size-1] float32
        input_row_offsets : [batch_size + 1]                      uint32

    Args:
        qkv_input_ragged: Flat projected QKV, all sequences concatenated.
        conv_weight: Depthwise conv weights, [conv_dim, kernel_size].
        conv_state_in: Initial sliding-window state, oldest slot first.
        input_row_offsets: uint32 exclusive prefix sums of sequence lengths.

    Returns:
        Tuple of:
          conv_output_ragged : [total_seq_len, conv_dim]             float32
          conv_state_out     : [batch_size, conv_dim, kernel_size-1] float32
    """
    device = qkv_input_ragged.device
    total_seq_len = qkv_input_ragged.shape[0]
    conv_dim = qkv_input_ragged.shape[1]

    conv_output_ragged_type = TensorType(
        DType.float32, [total_seq_len, conv_dim], device
    )
    conv_state_out_type = TensorType(DType.float32, conv_state_in.shape, device)

    # input_row_offsets must be uint32 for the Mojo op
    offsets_uint32 = (
        input_row_offsets
        if input_row_offsets.type.dtype == DType.uint32
        else input_row_offsets.cast(DType.uint32)
    )

    results = ops.custom(
        "gated_delta_conv1d_fwd",
        device,
        [qkv_input_ragged, conv_weight, conv_state_in, offsets_uint32],
        [conv_output_ragged_type, conv_state_out_type],
    )
    return cast(TensorValue, results[0]), cast(TensorValue, results[1])


def gated_delta_recurrence_fwd(
    qkv_conv_output: TensorValue,
    decay_per_token: TensorValue,
    beta_per_token: TensorValue,
    recurrent_state_in: TensorValue,
    input_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Pass 2: gated delta rule recurrence over conv1d outputs for a ragged batch.

    Each GPU thread handles one (batch_item, value_head, value_dim_element)
    triple and iterates sequentially over its sequence tokens.  The KD-element
    recurrent state column lives in registers, so there is no shared-memory
    traffic for the state.

    All preprocessing (L2 normalisation of Q/K, Q scaling, GQA head expansion)
    is fused inside the kernel.

    Tensor shapes:
        qkv_conv_output    : [total_seq_len, conv_dim]              float32
            Conv1d output from Pass 1.  Channel layout:
              Q at [0, key_dim), K at [key_dim, 2*key_dim),
              V at [2*key_dim, 2*key_dim + value_dim).
        decay_per_token    : [total_seq_len, num_value_heads]       float32
            Per-token per-head decay (exp(-softplus) pre-applied).
        beta_per_token     : [total_seq_len, num_value_heads]       float32
            Per-token per-head beta gate (sigmoid pre-applied).
        recurrent_state_in : [batch_size, nv, key_head_dim, value_head_dim]
            Initial recurrent KV memory.
        input_row_offsets  : [batch_size + 1]                      uint32

    Args:
        qkv_conv_output: Causal conv1d output from gated_delta_conv1d_fwd.
        decay_per_token: Decay scalars, already exp(-softplus) applied.
        beta_per_token: Beta gate values, already sigmoid applied.
        recurrent_state_in: Initial KV memory, shape [B, nv, kd, vd].
        input_row_offsets: uint32 exclusive prefix sums of sequence lengths.

    Returns:
        Tuple of:
          recurrence_output   : [total_seq_len, value_dim]            float32
          recurrent_state_out : [batch_size, nv, key_head_dim, value_head_dim]
    """
    device = qkv_conv_output.device
    total_seq_len = qkv_conv_output.shape[0]
    num_value_heads = decay_per_token.shape[1]
    value_head_dim = recurrent_state_in.shape[3]
    value_dim = num_value_heads * value_head_dim

    recurrence_output_type = TensorType(
        DType.float32, [total_seq_len, value_dim], device
    )
    recurrent_state_out_type = TensorType(
        DType.float32, recurrent_state_in.shape, device
    )

    # input_row_offsets must be uint32 for the Mojo op
    offsets_uint32 = (
        input_row_offsets
        if input_row_offsets.type.dtype == DType.uint32
        else input_row_offsets.cast(DType.uint32)
    )

    results = ops.custom(
        "gated_delta_recurrence_fwd",
        device,
        [
            qkv_conv_output,
            decay_per_token,
            beta_per_token,
            recurrent_state_in,
            offsets_uint32,
        ],
        [recurrence_output_type, recurrent_state_out_type],
    )
    return cast(TensorValue, results[0]), cast(TensorValue, results[1])
