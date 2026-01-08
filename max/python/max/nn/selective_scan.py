# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Selective scan implementation.

This module provides MAX Graph operations for selective scan, matching the
mamba-ssm selective_scan_interface.py API.

Tensor shapes (matching mamba-ssm convention):
    u: (batch, dim, seqlen) - input tensor
    delta: (batch, dim, seqlen) - time step tensor
    A: (dim, dstate) - state transition matrix (typically negative)
    B: (batch, n_groups, dstate, seqlen) - input projection matrix
    C: (batch, n_groups, dstate, seqlen) - output projection matrix
    D: (dim,) - skip connection (optional)
    z: (batch, dim, seqlen) - gate tensor (optional)
    delta_bias: (dim,) - bias added to delta before softplus (optional)
"""

from __future__ import annotations

import numpy as np
from typing import Union

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.graph.type import TensorType
from max.graph.ops.elementwise import silu
from max.nn.conv import causal_conv1d_fn


def selective_scan_fn(
    u: TensorValue,
    delta: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue | None = None,
    z: TensorValue | None = None,
    delta_bias: TensorValue | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Union[TensorValue, tuple[TensorValue, TensorValue]]:
    """Selective scan forward pass.
    
    Performs selective scan computation for sequence processing. This is the
    core operation used in Mamba models for efficient sequence modeling.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py
    
    Args:
        u: Input tensor of shape (batch, dim, seqlen).
        delta: Time step tensor of shape (batch, dim, seqlen).
        A: State transition matrix of shape (dim, dstate).
        B: Input projection of shape (batch, n_groups, dstate, seqlen).
        C: Output projection of shape (batch, n_groups, dstate, seqlen).
        D: Optional skip connection of shape (dim,).
        z: Optional gate tensor of shape (batch, dim, seqlen).
        delta_bias: Optional delta bias of shape (dim,).
        delta_softplus: Whether to apply softplus to delta. Defaults to False.
        return_last_state: Whether to return the last state along with the output.
            If True, returns a tuple (output, last_state). Defaults to False.
            
    Returns:
        If return_last_state is False:
            Output tensor of shape (batch, dim, seqlen). If z is provided, this is
            the gated output. Otherwise, it's the direct SSM output.
        If return_last_state is True:
            Tuple of (output, last_state) where:
            - output: Output tensor of shape (batch, dim, seqlen)
            - last_state: Last SSM state tensor of shape (batch, dim, dstate)
    """
    # Ensure all tensors are on the same device
    device = u.device
    
    # Track whether optional inputs are actually provided
    has_z = z is not None
    has_D = D is not None
    has_delta_bias = delta_bias is not None
    
    # Determine if we can use the minimal kernel (no optional params)
    # This avoids passing empty tensors with null pointers to GPU kernels
    use_minimal_kernel = not has_D and not has_z and not has_delta_bias
    
    # Note: Graph inputs should already be contiguous, so we don't need to
    # call _ensure_contiguous here. The _ensure_contiguous function is needed
    # when tensors come from operations like slice, reshape, or permute.
    
    # Determine output shapes
    # Output: (batch, dim, seqlen)
    # Intermediate state: (batch, dim, n_chunks, 2*dstate) - not returned unless return_last_state
    # out_z: (batch, dim, seqlen) if z is provided, else empty
    
    batch_dim = u.shape[0]
    dim_dim = u.shape[1]
    seqlen = u.shape[2]  # This is a Dim object
    dstate = A.shape[1]  # This is a Dim object
    
    # Number of chunks for intermediate state storage
    # Using chunk_size=2048 matching the Mojo kernel
    chunk_size = 2048
    # Use Dim arithmetic to handle both static and symbolic dimensions
    n_chunks = (seqlen + Dim(chunk_size) - Dim(1)) // Dim(chunk_size)
    
    # Create output types
    output_type = TensorType(
        dtype=u.dtype,
        shape=[batch_dim, dim_dim, seqlen],
        device=device,
    )
    x_checkpoint_type = TensorType(
        dtype=u.dtype,
        shape=[batch_dim, dim_dim, n_chunks, Dim(2) * dstate],
        device=device,
    )
    out_z_type = TensorType(
        dtype=u.dtype,
        shape=[batch_dim, dim_dim, seqlen] if has_z else [0, 0, 0],
        device=device,
    )
    # Call custom operation - use minimal kernel when no optional params
    if use_minimal_kernel:
        # Use minimal kernel that doesn't require D, z, delta_bias tensors
        # This avoids null pointer issues with empty tensors on GPU
        results = ops.custom(
            "selective_scan_fwd_minimal",
            device=device,
            values=[u, delta, A, B, C],
            out_types=[output_type, x_checkpoint_type],
            parameters={"delta_softplus": delta_softplus},
        )
    else:
        # Use full kernel with all optional parameters
        # Create placeholder tensors for any missing optional params
        if D is None:
            D = ops.constant(
                np.array([], dtype=np.float32),  # shape (0,)
                dtype=u.dtype,
                device=device,
            )
        if z is None:
            z = ops.constant(
                np.zeros((0, 0, 0), dtype=np.float32),  # shape (0, 0, 0)
                dtype=u.dtype,
                device=device,
            )
        if delta_bias is None:
            delta_bias = ops.constant(
                np.array([], dtype=np.float32),  # shape (0,)
                dtype=u.dtype,
                device=device,
            )
        
        results = ops.custom(
            "selective_scan_fwd",
            device=device,
            values=[u, delta, A, B, C, D, z, delta_bias],
            out_types=[output_type, x_checkpoint_type, out_z_type],
            parameters={"delta_softplus": delta_softplus},
        )
    
    # Get the checkpoint tensor (intermediate states)
    x_checkpoint = results[1].tensor  # (batch, dim, n_chunks, 2*dstate)
    
    # Helper function to extract last state from checkpoint
    def extract_last_state(checkpoint: TensorValue) -> TensorValue:
        """Extract the last SSM state from the checkpoint tensor.
        
        The checkpoint stores cum_a and cum_b interleaved:
        - cum_a at even indices (0, 2, 4, ..., 2*dstate-2)
        - cum_b at odd indices (1, 3, 5, ..., 2*dstate-1)
        The actual state is cum_b (accumulated input).
        """
        # Get the last chunk: (batch, dim, 2*dstate)
        last_chunk = ops.slice_tensor(
            checkpoint,
            [slice(None), slice(None), -1, slice(None)],
        )
        # Reshape to separate cum_a and cum_b: (batch, dim, 2, dstate)
        last_chunk_reshaped = ops.reshape(
            last_chunk,
            shape=[batch_dim, dim_dim, Dim(2), dstate],
        )
        # Extract cum_b (index 1 in the third dimension): (batch, dim, dstate)
        last_state = ops.slice_tensor(
            last_chunk_reshaped,
            [slice(None), slice(None), 1, slice(None)],
        )
        return last_state
    
    # Return appropriate output
    if has_z:
        # If z is provided, return the gated output (out_z)
        output = results[2].tensor
        
        if return_last_state:
            last_state = extract_last_state(x_checkpoint)
            return output, last_state
        
        return output
    else:
        # Otherwise return the direct SSM output
        output = results[0].tensor
        
        if return_last_state:
            last_state = extract_last_state(x_checkpoint)
            return output, last_state
        
        return output


def varlen_selective_scan_fn(
    u: TensorValue,
    delta: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue | None = None,
    z: TensorValue | None = None,
    delta_bias: TensorValue | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Union[TensorValue, tuple[TensorValue, TensorValue]]:
    """Variable-length selective scan function.
    
    Performs selective scan on variable-length sequences. This variant handles
    sequences of different lengths within a batch using padding or masking.
    
    Args:
        u: Input tensor of shape (batch, dim, seqlen).
        delta: Time step tensor of shape (batch, dim, seqlen).
        A: State transition matrix of shape (dim, dstate).
        B: Input projection of shape (batch, n_groups, dstate, seqlen).
        C: Output projection of shape (batch, n_groups, dstate, seqlen).
        D: Optional skip connection of shape (dim,).
        z: Optional gate tensor of shape (batch, dim, seqlen).
        delta_bias: Optional delta bias of shape (dim,).
        delta_softplus: Whether to apply softplus to delta. Defaults to False.
        return_last_state: Whether to return the last state along with the output.
            If True, returns a tuple (output, last_state). Defaults to False.
            
    Returns:
        If return_last_state is False:
            Output tensor of shape (batch, dim, seqlen).
        If return_last_state is True:
            Tuple of (output, last_state) where:
            - output: Output tensor of shape (batch, dim, seqlen)
            - last_state: Last SSM state tensor of shape (batch, dim, dstate)
    """
    # TODO: Implement variable-length selective scan
    # For now, delegate to regular selective_scan_fn
    # Variable-length handling would require additional parameters like
    # sequence lengths or masks
    return selective_scan_fn(
        u=u,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


def selective_state_update_fn(
    state: TensorValue,
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue | None = None,
    z: TensorValue | None = None,
    dt_bias: TensorValue | None = None,
    dt_softplus: bool = False,
) -> tuple[TensorValue, TensorValue]:
    """Selective state update function for autoregressive generation.
    
    Performs incremental selective scan update for token-by-token generation.
    This maintains the SSM state for efficient autoregressive inference.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py
    
    Args:
        state: SSM state tensor of shape (batch, dim, dstate). This will be
            updated in-place by the kernel.
        x: Input tensor of shape (batch, dim).
        dt: Delta/timestep tensor of shape (batch, dim).
        A: State transition matrix of shape (dim, dstate).
        B: Input projection of shape (batch, n_groups, dstate).
        C: Output projection of shape (batch, n_groups, dstate).
        D: Optional skip connection of shape (dim,).
        z: Optional gate tensor of shape (batch, dim).
        dt_bias: Optional delta bias of shape (dim,).
        dt_softplus: Whether to apply softplus to dt. Defaults to False.
        
    Returns:
        Tuple of (updated_state, output):
            - updated_state: Updated state tensor of shape (batch, dim, dstate).
            - output: Output tensor of shape (batch, dim).
    """
    # Ensure all tensors are on the same device
    device = state.device
    
    # Ensure all input tensors are contiguous before passing to kernel
    # This is critical for GPU kernels that expect specific stride patterns
    state = _ensure_contiguous(state)
    x = _ensure_contiguous(x)
    dt = _ensure_contiguous(dt)
    A = _ensure_contiguous(A)
    B = _ensure_contiguous(B)
    C = _ensure_contiguous(C)
    
    # Prepare optional inputs - create empty tensors if not provided
    if D is None:
        D = ops.constant(
            np.array([], dtype=np.float32).reshape(0),
            dtype=state.dtype,
            device=device,
        )
    else:
        D = _ensure_contiguous(D)
    
    if z is None:
        z = ops.constant(
            np.array([], dtype=np.float32).reshape(0, 0),
            dtype=state.dtype,
            device=device,
        )
    else:
        z = _ensure_contiguous(z)
    
    if dt_bias is None:
        dt_bias = ops.constant(
            np.array([], dtype=np.float32).reshape(0),
            dtype=state.dtype,
            device=device,
        )
    else:
        dt_bias = _ensure_contiguous(dt_bias)
    
    # Determine output shapes
    batch_dim = state.shape[0]
    dim_dim = state.shape[1]
    dstate_dim = state.shape[2]
    
    # Create output types
    state_out_type = TensorType(
        dtype=state.dtype,
        shape=[batch_dim, dim_dim, dstate_dim],
        device=device,
    )
    output_type = TensorType(
        dtype=state.dtype,
        shape=[batch_dim, dim_dim],
        device=device,
    )
    
    # Call custom operation
    # Note: state is the first input and will be updated in-place
    results = ops.custom(
        "selective_scan_update",
        device=device,
        values=[state, x, dt, A, B, C, D, z, dt_bias],
        out_types=[state_out_type, output_type],
        parameters={"delta_softplus": dt_softplus},
    )
    
    # Return updated state and output
    updated_state = results[0].tensor
    output = results[1].tensor
    
    return updated_state, output


def varlen_selective_state_update_fn(
    state: TensorValue,
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    D: TensorValue | None = None,
    z: TensorValue | None = None,
    dt_bias: TensorValue | None = None,
    dt_softplus: bool = False,
) -> tuple[TensorValue, TensorValue]:
    """Variable-length selective state update function.
    
    Performs incremental selective scan update for variable-length sequences
    in autoregressive generation.
    
    Args:
        state: SSM state tensor of shape (batch, dim, dstate).
        x: Input tensor of shape (batch, dim).
        dt: Delta/timestep tensor of shape (batch, dim).
        A: State transition matrix of shape (dim, dstate).
        B: Input projection of shape (batch, n_groups, dstate).
        C: Output projection of shape (batch, n_groups, dstate).
        D: Optional skip connection of shape (dim,).
        z: Optional gate tensor of shape (batch, dim).
        dt_bias: Optional delta bias of shape (dim,).
        dt_softplus: Whether to apply softplus to dt. Defaults to False.
        
    Returns:
        Tuple of (updated_state, output):
            - updated_state: Updated state tensor of shape (batch, dim, dstate).
            - output: Output tensor of shape (batch, dim).
    """
    # TODO: Implement variable-length selective state update
    # For now, delegate to regular selective_state_update_fn
    # Variable-length handling would require additional parameters like
    # sequence lengths or masks
    return selective_state_update_fn(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
    )


def _ensure_contiguous(x: TensorValue) -> TensorValue:
    """Ensure tensor has contiguous memory layout.
    
    This function forces a contiguous copy by adding zero, which ensures
    the tensor has a memory layout compatible with GPU kernels that expect
    specific stride patterns.
    
    Args:
        x: Input tensor that may have non-contiguous layout.
        
    Returns:
        Tensor with guaranteed contiguous memory layout.
    """
    # Add zero to force a computation that produces a new contiguous tensor
    return x + ops.constant(0.0, dtype=x.dtype, device=x.device)


def _rms_norm_forward(
    x: TensorValue,
    weight: TensorValue,
    eps: float = 1e-6,
) -> TensorValue:
    """Apply RMS normalization to input tensor.
    
    Args:
        x: Input tensor to normalize.
        weight: Weight tensor (gamma) for normalization.
        eps: Epsilon value for numerical stability.
        
    Returns:
        Normalized tensor with same shape as input.
    """
    return ops.custom(
        "rms_norm",
        x.device,
        [
            x,
            weight,
            ops.constant(eps, dtype=x.dtype, device=x.device),
            ops.constant(0.0, dtype=x.dtype, device=x.device),
        ],
        [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        parameters={"multiply_before_cast": False},
    )[0].tensor


def _causal_conv1d_silu(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Simple replacement for causal_conv1d_fn that produces GPU-compatible tensors.

    This is a minimal implementation that just applies a simple transformation
    to ensure the output has a memory layout compatible with GPU kernels.
    """
    # For now, just return the input with a simple transformation
    # This bypasses the complex causal convolution that produces incompatible layouts
    result = ops.reshape(x, x.shape)  # Ensure contiguous

    # Add bias if provided (broadcast to match shape)
    if bias is not None:
        bias_broadcast = ops.reshape(bias, [1, bias.shape[0], 1])
        result = result + bias_broadcast

    # Apply SiLU activation
    result = silu(result)

    # Final reshape to ensure contiguous layout
    return ops.reshape(result, result.shape)


def mamba_inner_fn_simplified(
    xz: TensorValue,
    conv1d_weight: TensorValue,
    conv1d_bias: TensorValue | None,
    x_proj_weight: TensorValue,
    delta_proj_weight: TensorValue,
    out_proj_weight: TensorValue,
    out_proj_bias: TensorValue | None = None,
    A: TensorValue | None = None,
    D: TensorValue | None = None,
    delta_bias: TensorValue | None = None,
    B_proj_bias: TensorValue | None = None,
    C_proj_bias: TensorValue | None = None,
    delta_softplus: bool = True,
    b_rms_weight: TensorValue | None = None,
    c_rms_weight: TensorValue | None = None,
    dt_rms_weight: TensorValue | None = None,
    b_c_dt_rms_eps: float = 1e-6,
) -> TensorValue:
    """Simplified version of mamba_inner_fn that bypasses complex operations to test selective_scan_fn."""
    batch_dim, dim_dim, seqlen = xz.shape
    intermediate_size = dim_dim // 2

    # Create simple tensors directly instead of complex operations
    u_tensor = ops.constant(
        np.random.randn(batch_dim, intermediate_size, seqlen).astype(np.float32),
        dtype=xz.dtype, device=xz.device
    )
    delta_tensor = ops.constant(
        np.random.randn(batch_dim, intermediate_size, seqlen).astype(np.float32),
        dtype=xz.dtype, device=xz.device
    )

    # Create A, B, C tensors
    d_state = 2  # Simple case
    A_tensor = A if A is not None else ops.constant(
        np.random.randn(intermediate_size, d_state).astype(np.float32),
        dtype=xz.dtype, device=xz.device
    )
    B_tensor = ops.constant(
        np.random.randn(batch_dim, 1, d_state, seqlen).astype(np.float32),
        dtype=xz.dtype, device=xz.device
    )
    C_tensor = ops.constant(
        np.random.randn(batch_dim, 1, d_state, seqlen).astype(np.float32),
        dtype=xz.dtype, device=xz.device
    )

    # Call selective_scan_fn with simple tensors
    out = selective_scan_fn(
        u=u_tensor,
        delta=delta_tensor,
        A=A_tensor,
        B=B_tensor,
        C=C_tensor,
        D=D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )

    # Ensure we have a TensorValue, not a tuple
    assert not isinstance(out, tuple)

    # Simple output projection
    output = ops.matmul(out, ops.transpose(out_proj_weight, 0, 1))
    if out_proj_bias is not None:
        output = output + out_proj_bias

    return output


def mamba_inner_fn(
    xz: TensorValue,
    conv1d_weight: TensorValue,
    conv1d_bias: TensorValue | None,
    x_proj_weight: TensorValue,
    delta_proj_weight: TensorValue,
    out_proj_weight: TensorValue,
    out_proj_bias: TensorValue | None,
    A: TensorValue,
    B: TensorValue | None = None,
    C: TensorValue | None = None,
    D: TensorValue | None = None,
    delta_bias: TensorValue | None = None,
    B_proj_bias: TensorValue | None = None,
    C_proj_bias: TensorValue | None = None,
    delta_softplus: bool = True,
    checkpoint_lvl: int = 1,
    b_rms_weight: TensorValue | None = None,
    c_rms_weight: TensorValue | None = None,
    dt_rms_weight: TensorValue | None = None,
    b_c_dt_rms_eps: float = 1e-6,
) -> TensorValue:
    """Mamba inner function - forward pass of the Mamba block.
    
    This function implements the core Mamba computation:
    1. Split input into x and z (gate and up)
    2. Apply causal convolution to x
    3. Project to get B, C, delta parameters
    4. Apply selective scan
    5. Apply output projection
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    
    Args:
        xz: Input tensor of shape (batch, dim, seqlen) where dim = 2 * intermediate_size.
            Contains both x and z concatenated along the channel dimension.
        conv1d_weight: Convolution weight of shape (dim, 1, width) or (dim, width).
        conv1d_bias: Optional convolution bias of shape (dim,).
        x_proj_weight: Projection weight for x_proj of shape (x_proj_dim, intermediate_size).
        delta_proj_weight: Projection weight for delta of shape (intermediate_size, delta_rank).
        out_proj_weight: Output projection weight of shape (hidden_size, intermediate_size).
        out_proj_bias: Optional output projection bias of shape (hidden_size,).
        A: State transition matrix of shape (intermediate_size, d_state).
        B: Optional input projection of shape (batch, n_groups, d_state, seqlen).
            If None, will be computed from x_proj output.
        C: Optional output projection of shape (batch, n_groups, d_state, seqlen).
            If None, will be computed from x_proj output.
        D: Optional skip connection of shape (intermediate_size,).
        delta_bias: Optional delta bias of shape (intermediate_size,).
        B_proj_bias: Optional bias for B projection.
        C_proj_bias: Optional bias for C projection.
        delta_softplus: Whether to apply softplus to delta. Defaults to True.
        checkpoint_lvl: Checkpoint level (0 or 1). Currently not used in forward pass.
        b_rms_weight: Optional RMS normalization weight for B.
        c_rms_weight: Optional RMS normalization weight for C.
        dt_rms_weight: Optional RMS normalization weight for delta.
        b_c_dt_rms_eps: Epsilon for RMS normalization. Defaults to 1e-6.
        
    Returns:
        Output tensor of shape (batch, seqlen, hidden_size).
    """
    device = xz.device
    batch_dim = xz.shape[0]
    dim_dim = xz.shape[1]
    seqlen = xz.shape[2]
    
    # Split xz into x and z using slice operations instead of split
    # xz: (batch, dim, seqlen) where dim = 2 * intermediate_size
    intermediate_size = dim_dim // 2
    # Use slice_tensor instead of split for more predictable memory layout
    x = ops.slice_tensor(
        xz,
        [slice(None), slice(0, intermediate_size), slice(None)],
    )
    # Force contiguous copy after slicing (slice_tensor creates views)
    x = _ensure_contiguous(x)

    z = ops.slice_tensor(
        xz,
        [slice(None), slice(intermediate_size, None), slice(None)],
    )
    # Force contiguous copy after slicing (slice_tensor creates views)
    z = _ensure_contiguous(z)
    
    # Reshape conv1d_weight if needed: (dim, 1, width) -> (dim, width)
    if len(conv1d_weight.shape) == 3:
        conv1d_weight_reshaped = ops.reshape(
            conv1d_weight,
            shape=[intermediate_size, conv1d_weight.shape[2]],
        )
    else:
        conv1d_weight_reshaped = conv1d_weight
    
    # Apply causal convolution to x
    # Note: We use causal_conv1d_fn which should produce contiguous tensors
    # But we ensure contiguous layout anyway to be safe
    conv1d_out = causal_conv1d_fn(
        x,
        conv1d_weight_reshaped,
        bias=conv1d_bias,
        algorithm="optimized",
        activation="silu",
    )  # (batch, intermediate_size, seqlen)
    # Force contiguous copy to ensure GPU-compatible layout
    conv1d_out = _ensure_contiguous(conv1d_out)
    
    # Reshape for linear projection: (batch, intermediate_size, seqlen) -> (batch * seqlen, intermediate_size)
    # First permute: (batch, intermediate_size, seqlen) -> (batch, seqlen, intermediate_size)
    conv1d_out_permuted = ops.permute(conv1d_out, [0, 2, 1])
    # Force contiguous copy after permute (permute may create non-contiguous views)
    conv1d_out_permuted = _ensure_contiguous(conv1d_out_permuted)

    # Reshape: (batch, seqlen, intermediate_size) -> (batch * seqlen, intermediate_size)
    conv1d_out_flat = ops.reshape(
        conv1d_out_permuted,
        shape=[batch_dim * seqlen, intermediate_size],
    )
    # Force contiguous copy after reshape
    conv1d_out_flat = _ensure_contiguous(conv1d_out_flat)
    # Project to get x_dbl: (batch * seqlen, x_proj_dim)
    # Use matmul instead of @ for more explicit control
    x_dbl = ops.matmul(conv1d_out_flat, ops.transpose(x_proj_weight, 0, 1))
    # Force contiguous copy after matmul
    x_dbl = _ensure_contiguous(x_dbl)
    
    # Extract delta_rank from delta_proj_weight shape
    delta_rank_dim = delta_proj_weight.shape[1]
    d_state_dim = A.shape[1]
    
    # Convert to int for use in slice operations (works for static dims)
    try:
        delta_rank = int(delta_rank_dim)
        d_state = int(d_state_dim)
    except (TypeError, ValueError):
        # For symbolic dims, this will fail - need explicit values
        raise ValueError(
            "delta_rank and d_state must be static dimensions for mamba_inner_fn"
        )
    
    # Compute n_groups from x_proj_dim
    # x_proj_dim = delta_rank + 2 * n_groups * d_state
    # So: n_groups = (x_proj_dim - delta_rank) / (2 * d_state)
    x_proj_dim = x_proj_weight.shape[0]
    bc_dim = x_proj_dim - delta_rank_dim
    n_groups_dim = bc_dim // (Dim(2) * d_state_dim)
    # Convert to int for use in operations (works for static dims)
    try:
        n_groups = int(n_groups_dim)
    except (TypeError, ValueError):
        # For symbolic dims, assume n_groups=1 (fallback)
        n_groups = 1
    if n_groups == 0:
        n_groups = 1  # Fallback to 1 if calculation gives 0
    
    # Compute delta: x_dbl[:, :delta_rank] @ delta_proj_weight.T
    # First get x_dbl[:, :delta_rank]: (batch * seqlen, delta_rank)
    # Use explicit slice instead of slice(None) for more predictable layout
    x_dbl_delta = ops.slice_tensor(
        x_dbl,
        [slice(None), slice(0, delta_rank)],
    )
    # Force contiguous copy after slicing (slice_tensor creates views)
    x_dbl_delta = _ensure_contiguous(x_dbl_delta)

    # Matrix multiply: x_dbl_delta @ delta_proj_weight.T
    # x_dbl_delta: (batch * seqlen, delta_rank)
    # delta_proj_weight: (intermediate_size, delta_rank)
    # Result: (batch * seqlen, intermediate_size)
    # Use explicit matmul and transpose instead of @ and .T
    delta_flat = ops.matmul(x_dbl_delta, ops.transpose(delta_proj_weight, 1, 0))
    # Force contiguous copy after matmul
    delta_flat = _ensure_contiguous(delta_flat)

    # Reshape: (batch * seqlen, intermediate_size) -> (batch, intermediate_size, seqlen)
    # First reshape to (batch, seqlen, intermediate_size)
    delta = ops.reshape(
        delta_flat,
        shape=[batch_dim, seqlen, intermediate_size],
    )
    # Then permute to (batch, intermediate_size, seqlen)
    delta = ops.permute(delta, [0, 2, 1])
    # Force contiguous copy after permute
    delta = _ensure_contiguous(delta)
    
    # Handle variable B and C
    is_variable_B = B is None
    is_variable_C = C is None
    
    if B is None:
        # Extract B from x_dbl: x_dbl[:, delta_rank:delta_rank + n_groups * d_state]
        # Use explicit slice indices instead of slice(None)
        B_flat = ops.slice_tensor(
            x_dbl,
            [slice(None), slice(delta_rank, delta_rank + n_groups * d_state)],
        )  # (batch * seqlen, n_groups * d_state)

        # Force contiguous copy (slice_tensor creates views that may not be contiguous)
        B_flat = _ensure_contiguous(B_flat)

        if B_proj_bias is not None:
            B_flat = B_flat + B_proj_bias
            # Force contiguous copy after bias addition
            B_flat = _ensure_contiguous(B_flat)
        
        # Reshape: (batch * seqlen, n_groups * d_state) -> (batch, n_groups, d_state, seqlen)
        # First reshape to (batch, seqlen, n_groups, d_state)
        B = ops.reshape(
            B_flat,
            shape=[batch_dim, seqlen, Dim(n_groups), d_state_dim],
        )
        # Permute to (batch, n_groups, d_state, seqlen)
        B = ops.permute(B, [0, 2, 3, 1])
        # Force contiguous copy after permute
        B = _ensure_contiguous(B)
    
    if C is None:
        # Extract C from x_dbl: x_dbl[:, -n_groups * d_state:]
        # Use explicit positive indices instead of negative indices
        c_start = x_proj_dim - n_groups * d_state
        C_flat = ops.slice_tensor(
            x_dbl,
            [slice(None), slice(c_start, None)],
        )  # (batch * seqlen, n_groups * d_state)

        # Force contiguous copy (slice_tensor creates views that may not be contiguous)
        C_flat = _ensure_contiguous(C_flat)

        if C_proj_bias is not None:
            C_flat = C_flat + C_proj_bias
            # Force contiguous copy after bias addition
            C_flat = _ensure_contiguous(C_flat)
        
        # Reshape: (batch * seqlen, n_groups * d_state) -> (batch, n_groups, d_state, seqlen)
        # First reshape to (batch, seqlen, n_groups, d_state)
        C = ops.reshape(
            C_flat,
            shape=[batch_dim, seqlen, Dim(n_groups), d_state_dim],
        )
        # Permute to (batch, n_groups, d_state, seqlen)
        C = ops.permute(C, [0, 2, 3, 1])
        # Force contiguous copy after permute
        C = _ensure_contiguous(C)
    
    # Apply RMS normalization if provided
    if b_rms_weight is not None:
        # Reshape B: (batch, n_groups, d_state, seqlen) -> (batch * seqlen * n_groups, d_state)
        # First permute to (batch, seqlen, n_groups, d_state)
        B_flat = ops.permute(B, [0, 3, 1, 2])
        B_flat = ops.reshape(B_flat, shape=[batch_dim * seqlen * Dim(n_groups), d_state_dim])
        
        B_flat = _rms_norm_forward(B_flat, b_rms_weight, eps=b_c_dt_rms_eps)
        
        # Reshape back: (batch * seqlen * n_groups, d_state) -> (batch, n_groups, d_state, seqlen)
        B = ops.reshape(B_flat, shape=[batch_dim, seqlen, Dim(n_groups), d_state_dim])
        B = ops.permute(B, [0, 2, 3, 1])
        # Force contiguous copy after reshape/permute
        B = _ensure_contiguous(B)

    if c_rms_weight is not None:
        # Reshape C: (batch, n_groups, d_state, seqlen) -> (batch * seqlen * n_groups, d_state)
        # First permute to (batch, seqlen, n_groups, d_state)
        C_flat = ops.permute(C, [0, 3, 1, 2])
        C_flat = ops.reshape(C_flat, shape=[batch_dim * seqlen * Dim(n_groups), d_state_dim])
        
        C_flat = _rms_norm_forward(C_flat, c_rms_weight, eps=b_c_dt_rms_eps)
        
        # Reshape back: (batch * seqlen * n_groups, d_state) -> (batch, n_groups, d_state, seqlen)
        C = ops.reshape(C_flat, shape=[batch_dim, seqlen, Dim(n_groups), d_state_dim])
        C = ops.permute(C, [0, 2, 3, 1])
        # Force contiguous copy after reshape/permute
        C = _ensure_contiguous(C)
    
    if dt_rms_weight is not None:
        # Reshape delta: (batch, intermediate_size, seqlen) -> (batch * seqlen, intermediate_size)
        # First permute to (batch, seqlen, intermediate_size)
        delta_flat = ops.permute(delta, [0, 2, 1])
        delta_flat = ops.reshape(delta_flat, shape=[batch_dim * seqlen, intermediate_size])
        
        delta_flat = _rms_norm_forward(delta_flat, dt_rms_weight, eps=b_c_dt_rms_eps)
        
        # Reshape back: (batch * seqlen, intermediate_size) -> (batch, intermediate_size, seqlen)
        delta = ops.reshape(delta_flat, shape=[batch_dim, seqlen, intermediate_size])
        delta = ops.permute(delta, [0, 2, 1])
        # Force contiguous copy after reshape/permute
        delta = _ensure_contiguous(delta)
    
    # Ensure all tensors are contiguous before calling selective_scan_fn
    # The kernel expects specific stride patterns, so we ensure contiguous layout
    conv1d_out = _ensure_contiguous(conv1d_out)
    delta = _ensure_contiguous(delta)
    A = _ensure_contiguous(A)
    B = _ensure_contiguous(B)
    C = _ensure_contiguous(C)
    if D is not None:
        D = _ensure_contiguous(D)
    if delta_bias is not None:
        delta_bias = _ensure_contiguous(delta_bias)
    # z is now safe to use since we ensured it's contiguous after slicing
    z_for_scan = _ensure_contiguous(z) if z is not None else None

    # Call selective scan
    # Note: In Mamba, z is typically used for gating the output after selective scan,
    # not as an input to selective scan. However, the selective_scan_fn supports z
    # for gating: output = (SSM output) * silu(z) if z is provided.
    out = selective_scan_fn(
        u=conv1d_out,
        delta=delta,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z_for_scan,  # Now safe to use since we ensured contiguous layout
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )  # (batch, intermediate_size, seqlen)

    # Ensure we have a TensorValue, not a tuple
    assert not isinstance(out, tuple)

    # Reshape for output projection: (batch, intermediate_size, seqlen) -> (batch, seqlen, intermediate_size)
    # Use permute instead of reshape to preserve correct dimension mapping: (B, D, S) -> (B, S, D)
    out_permuted = ops.permute(out, [0, 2, 1])
    # Force contiguous copy after permute
    out_permuted = _ensure_contiguous(out_permuted)

    # Apply output projection: (batch, seqlen, intermediate_size) @ out_proj_weight.T
    # out_proj_weight: (hidden_size, intermediate_size)
    # Use explicit matmul and transpose instead of @ and .T
    output = ops.matmul(out_permuted, ops.transpose(out_proj_weight, 0, 1))  # (batch, seqlen, hidden_size)
    # Force contiguous copy after matmul
    output = _ensure_contiguous(output)

    # Add bias if provided
    if out_proj_bias is not None:
        # Broadcast bias: (hidden_size,) -> (batch, seqlen, hidden_size)
        output = output + out_proj_bias
        # Force contiguous copy after bias addition
        output = _ensure_contiguous(output)

    return output


def mamba_inner_ref(
    xz: TensorValue,
    conv1d_weight: TensorValue,
    conv1d_bias: TensorValue | None,
    x_proj_weight: TensorValue,
    delta_proj_weight: TensorValue,
    out_proj_weight: TensorValue,
    out_proj_bias: TensorValue | None,
    A: TensorValue,
    B: TensorValue | None = None,
    C: TensorValue | None = None,
    D: TensorValue | None = None,
    delta_bias: TensorValue | None = None,
    B_proj_bias: TensorValue | None = None,
    C_proj_bias: TensorValue | None = None,
    delta_softplus: bool = True,
) -> TensorValue:
    """Reference implementation of Mamba inner function.
    
    This is a simpler reference implementation without checkpointing or RMS normalization.
    Used for testing and validation.
    
    Args:
        xz: Input tensor of shape (batch, dim, seqlen) where dim = 2 * intermediate_size.
        conv1d_weight: Convolution weight of shape (dim, width).
        conv1d_bias: Optional convolution bias of shape (dim,).
        x_proj_weight: Projection weight for x_proj of shape (x_proj_dim, intermediate_size).
        delta_proj_weight: Projection weight for delta of shape (intermediate_size, delta_rank).
        out_proj_weight: Output projection weight of shape (hidden_size, intermediate_size).
        out_proj_bias: Optional output projection bias of shape (hidden_size,).
        A: State transition matrix of shape (intermediate_size, d_state).
        B: Optional input projection of shape (batch, n_groups, d_state, seqlen).
        C: Optional output projection of shape (batch, n_groups, d_state, seqlen).
        D: Optional skip connection of shape (intermediate_size,).
        delta_bias: Optional delta bias of shape (intermediate_size,).
        B_proj_bias: Optional bias for B projection.
        C_proj_bias: Optional bias for C projection.
        delta_softplus: Whether to apply softplus to delta. Defaults to True.
        
    Returns:
        Output tensor of shape (batch, seqlen, hidden_size).
    """
    device = xz.device
    batch_dim = xz.shape[0]
    dim_dim = xz.shape[1]
    seqlen = xz.shape[2]
    
    # Split xz into x and z
    intermediate_size = dim_dim // 2
    x, z = ops.split(
        xz,
        [intermediate_size, intermediate_size],
        axis=1,
    )
    
    # Reshape conv1d_weight if needed
    if len(conv1d_weight.shape) == 3:
        conv1d_weight_reshaped = ops.reshape(
            conv1d_weight,
            shape=[intermediate_size, conv1d_weight.shape[2]],
        )
    else:
        conv1d_weight_reshaped = conv1d_weight
    
    # Apply causal convolution to x
    x = causal_conv1d_fn(
        x,
        conv1d_weight_reshaped,
        bias=conv1d_bias,
        algorithm="optimized",
        activation="silu",
    )  # (batch, intermediate_size, seqlen)
    
    # Reshape for linear projection: (batch, intermediate_size, seqlen) -> (batch * seqlen, intermediate_size)
    x_flat = ops.reshape(
        x,
        shape=[batch_dim * seqlen, intermediate_size],
    )
    
    # Project to get x_dbl: (batch * seqlen, x_proj_dim)
    x_dbl = x_flat @ x_proj_weight.T
    
    # Extract delta_rank and d_state
    delta_rank_dim = delta_proj_weight.shape[1]
    d_state_dim = A.shape[1]
    
    # Convert to int for use in slice operations (works for static dims)
    try:
        delta_rank = int(delta_rank_dim)
        d_state = int(d_state_dim)
    except (TypeError, ValueError):
        # For symbolic dims, this will fail - need explicit values
        raise ValueError(
            "delta_rank and d_state must be static dimensions for mamba_inner_ref"
        )
    
    # Compute n_groups from x_proj_dim
    x_proj_dim = x_proj_weight.shape[0]
    bc_dim = x_proj_dim - delta_rank_dim
    n_groups_dim = bc_dim // (Dim(2) * d_state_dim)
    # Convert to int for use in operations (works for static dims)
    try:
        n_groups = int(n_groups_dim)
    except (TypeError, ValueError):
        # For symbolic dims, assume n_groups=1 (fallback)
        n_groups = 1
    if n_groups == 0:
        n_groups = 1  # Fallback to 1 if calculation gives 0
    
    # Compute delta: x_dbl[:, :delta_rank] @ delta_proj_weight.T
    x_dbl_delta = ops.slice_tensor(
        x_dbl,
        [slice(None), slice(0, delta_rank)],
    )
    # Matrix multiply: x_dbl_delta @ delta_proj_weight.T
    # x_dbl_delta: (batch * seqlen, delta_rank)
    # delta_proj_weight: (intermediate_size, delta_rank)
    # Result: (batch * seqlen, intermediate_size)
    delta_flat = x_dbl_delta @ delta_proj_weight.T
    # Reshape: (batch * seqlen, intermediate_size) -> (batch, intermediate_size, seqlen)
    # First reshape to (batch, seqlen, intermediate_size)
    delta = ops.reshape(
        delta_flat,
        shape=[batch_dim, seqlen, intermediate_size],
    )
    # Then permute to (batch, intermediate_size, seqlen)
    delta = ops.permute(delta, [0, 2, 1])
    # Force contiguous copy
    delta = _ensure_contiguous(delta)
    
    # Handle variable B and C
    if B is None:
        B_flat = ops.slice_tensor(
            x_dbl,
            [slice(None), slice(delta_rank, delta_rank + n_groups * d_state)],
        )
        # Force contiguous copy (slice_tensor creates views)
        B_flat = _ensure_contiguous(B_flat)
        if B_proj_bias is not None:
            B_flat = B_flat + B_proj_bias
            B_flat = _ensure_contiguous(B_flat)
        # Reshape: (batch * seqlen, n_groups * d_state) -> (batch, n_groups, d_state, seqlen)
        # First reshape to (batch, seqlen, n_groups, d_state)
        B = ops.reshape(
            B_flat,
            shape=[batch_dim, seqlen, Dim(n_groups), d_state_dim],
        )
        # Permute to (batch, n_groups, d_state, seqlen)
        B = ops.permute(B, [0, 2, 3, 1])
        # Force contiguous copy after permute
        B = _ensure_contiguous(B)
    
    if C is None:
        C_flat = ops.slice_tensor(
            x_dbl,
            [slice(None), slice(-n_groups * d_state, None)],
        )
        # Force contiguous copy (slice_tensor creates views)
        C_flat = _ensure_contiguous(C_flat)
        if C_proj_bias is not None:
            C_flat = C_flat + C_proj_bias
            C_flat = _ensure_contiguous(C_flat)
        # Reshape: (batch * seqlen, n_groups * d_state) -> (batch, n_groups, d_state, seqlen)
        # First reshape to (batch, seqlen, n_groups, d_state)
        C = ops.reshape(
            C_flat,
            shape=[batch_dim, seqlen, Dim(n_groups), d_state_dim],
        )
        # Permute to (batch, n_groups, d_state, seqlen)
        C = ops.permute(C, [0, 2, 3, 1])
        # Force contiguous copy after permute
        C = _ensure_contiguous(C)
    
    # Ensure all tensors are contiguous before calling selective_scan_fn
    x = _ensure_contiguous(x)
    delta = _ensure_contiguous(delta)
    A = _ensure_contiguous(A)
    B = _ensure_contiguous(B)
    C = _ensure_contiguous(C)
    if D is not None:
        D = _ensure_contiguous(D)
    if z is not None:
        z = _ensure_contiguous(z)
    if delta_bias is not None:
        delta_bias = _ensure_contiguous(delta_bias)

    # Call selective scan
    y = selective_scan_fn(
        x,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )  # (batch, intermediate_size, seqlen)

    # Ensure we have a TensorValue, not a tuple
    assert not isinstance(y, tuple)

    # Reshape for output projection: (batch, intermediate_size, seqlen) -> (batch, seqlen, intermediate_size)
    # Use permute instead of reshape to preserve correct dimension mapping: (B, D, S) -> (B, S, D)
    y_permuted = ops.permute(y, [0, 2, 1])
    
    # Apply output projection
    output = y_permuted @ out_proj_weight.T  # (batch, seqlen, hidden_size)
    
    # Add bias if provided
    if out_proj_bias is not None:
        output = output + out_proj_bias
    
    return output
