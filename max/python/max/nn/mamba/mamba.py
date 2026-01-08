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
"""Mamba block implementation based on the reference implementation.

Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, Weight, ops
from max.nn import Layer, Linear, Module
from max.nn.conv import causal_conv1d_fn, causal_conv1d_update_fn
from max.nn.norm import layer_norm_fn
from max.nn.selective_scan import mamba_inner_fn


class MambaSSM(Module):
    """State Space Model layer for Mamba.
    
    Implements the Mamba SSM layer based on the reference implementation:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    
    The layer performs:
    1. Input projection and gating (gate + up)
    2. Causal 1D convolution
    3. State space parameter projection (B, C, delta)
    4. Selective scan computation
    5. Output projection
    
    Note: Full implementation with selective_scan_fwd requires proper handling
    of ragged tensors and tensor reshaping. This is a foundational implementation
    that can be extended as kernel support improves.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: DType,
        device: DeviceRef,
        linear_cls: Callable[..., Linear],
        d_state: int = 16,
        dt_rank: str | int = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        delta_softplus: bool = True,
        conv_kernel: int = 4,
        x_proj_dim: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        """Initialize the Mamba SSM layer.
        
        Matches the reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            hidden_size: Hidden dimension size (d_model in reference).
            intermediate_size: Intermediate dimension for SSM expansion (d_inner in reference).
            dtype: Data type for weights.
            device: Device to place weights on.
            linear_cls: Linear layer class to use.
            d_state: State space dimension (default: 16).
            dt_rank: Rank of delta projection. "auto" sets it to ceil(hidden_size / 16).
            conv_bias: Whether to use bias in conv1d.
            bias: Whether to use bias in linear projections.
            delta_softplus: Whether to apply softplus to delta values (default: True).
            conv_kernel: Convolution kernel size (default: 4).
            x_proj_dim: Output dimension of x_proj (inferred from weights if None).
            dt_min: Minimum value for dt initialization (default: 0.001).
            dt_max: Maximum value for dt initialization (default: 0.1).
            dt_init: Initialization method for dt_proj weights ("random" or "constant").
            dt_scale: Scale factor for dt_proj weight initialization (default: 1.0).
            dt_init_floor: Floor value for dt initialization (default: 1e-4).
            use_fast_path: Whether to use fast fused kernel path when available (default: True).
            layer_idx: Layer index for inference caching (default: None).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.device = device
        self.d_state = d_state
        self.delta_softplus = delta_softplus
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        # Determine dt_rank - match reference: math.ceil(d_model / 16)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(hidden_size / 16)
        else:
            self.dt_rank = int(dt_rank)
        
        # Input projection: projects to intermediate_size * 2 (gate + up)
        self.in_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=intermediate_size * 2,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        
        # Conv1d layer for sequence processing
        # The conv1d processes the gated input with a causal convolution
        # Weight shape: (intermediate_size, conv_width)
        # Use conv_kernel from config
        self.conv_width = conv_kernel
        
        # Use Conv1D for the convolution
        # Note: Conv1D doesn't have a causal parameter, so we'll use causal_conv1d_fn
        # or handle causal padding manually
        # For now, create Conv1D and we'll use causal_conv1d_fn in forward
        self.conv1d_weight = Weight(
            name="conv1d_weight",
            dtype=dtype,
            shape=(intermediate_size, self.conv_width),  # (channels, width)
            device=device,
        )
        self.conv1d_bias: Weight | None = None
        if conv_bias:
            self.conv1d_bias = Weight(
                name="conv1d_bias",
                dtype=dtype,
                shape=(intermediate_size,),
                device=device,
            )
        
        # State space parameter projections
        # x_proj: projects conv output to (dt_rank, B, C) parameters
        # Reference: self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2)
        # B and C each have d_state elements, not grouped (n_groups=1 in standard Mamba)
        # 
        # Note: Some Mamba variants use n_groups > 1 for grouped B/C projections,
        # but standard Mamba (mamba_simple.py) uses n_groups=1.
        self.n_groups = 1  # Standard Mamba uses n_groups=1
        
        # x_proj_dim = dt_rank + d_state * 2 (for B and C, each of size d_state)
        # If x_proj_dim is provided from weights, use it and infer dt_rank
        if x_proj_dim is not None:
            # Infer dt_rank from provided x_proj_dim
            # x_proj_dim = dt_rank + 2 * d_state
            # So: dt_rank = x_proj_dim - 2 * d_state
            calculated_dt_rank = x_proj_dim - 2 * d_state
            if calculated_dt_rank > 0:
                self.dt_rank = calculated_dt_rank
            # else keep dt_rank from "auto" calculation above
        else:
            # Calculate x_proj_dim from dt_rank and d_state
            x_proj_dim = self.dt_rank + 2 * d_state
        
        self.x_proj = linear_cls(
            in_dim=intermediate_size,
            out_dim=x_proj_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        
        # Delta projection parameters
        self.dt_proj = linear_cls(
            in_dim=self.dt_rank,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=True,  # dt_bias
        )
        
        # State transition matrix A (intermediate_size, d_state)
        # A is initialized in log space and shared across all sequences
        # A_log is a learnable parameter, A = exp(A_log)
        # Shape: (intermediate_size, d_state)
        # 
        # Weight loading: This weight is automatically registered by Module.__setattr__
        # and will be loaded from state dict with name "layers.{i}.mixer.A_log"
        # when load_state_dict() is called. The weight should be in log space
        # (i.e., values should be log(A), not A itself).
        self.A_log = Weight(
            name="A_log",
            dtype=dtype,
            shape=(intermediate_size, d_state),
            device=device,
        )
        
        # Skip connection D (optional, shape: intermediate_size)
        # D is a learnable parameter for the skip connection
        # 
        # Weight loading: This weight is automatically registered by Module.__setattr__
        # and will be loaded from state dict with name "layers.{i}.mixer.D"
        # when load_state_dict() is called.
        self.D = Weight(
            name="D",
            dtype=dtype,
            shape=(intermediate_size,),
            device=device,
        )
        
        # Output projection
        self.out_proj = linear_cls(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        
        # State buffers for autoregressive generation
        # These will be initialized on first use
        # Reference shapes: conv_state (batch, d_inner, d_conv), ssm_state (batch, d_inner, d_state)
        self._conv_state: Weight | None = None  # (batch, intermediate_size, d_conv)
        self._ssm_state: Weight | None = None  # (batch, intermediate_size, d_state)

    def __call__(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue | None = None,
        inference_params: Optional[dict] = None,
        **kwargs,
    ) -> TensorValue:
        """Forward pass through the SSM layer.
        
        Args:
            x: Input hidden states of shape (batch * seqlen, hidden_size) or (batch, seqlen, hidden_size).
            input_row_offsets: Row offsets for ragged tensor processing.
            inference_params: Optional inference parameters for caching.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Output after SSM transformation.
        """
        # Step 1: Check if we should use step method (autoregressive mode)
        # Only use step when we *know* we have a single token (seqlen==1); otherwise fall back to prefill
        use_step = False
        if inference_params is not None and self.layer_idx is not None:
            seqlen_offset = inference_params.get("seqlen_offset", 0)
        else:
            seqlen_offset = 0

            def _is_one(dim: Dim | int) -> bool:
                try:
                    return int(dim) == 1
                except (TypeError, ValueError):
                    return False

            # Require concrete single-token inputs to enter step path
            use_step = seqlen_offset > 0 and _is_one(x.shape[0]) and (
                input_row_offsets is None or _is_one(input_row_offsets.shape[0] - 1)
            )
        
        # Infer batch and seqlen early for step method
        # For now, simplify and use batch_size=1 to avoid ragged tensor complexity
        batch_size = 1
        seqlen = x.shape[0]
        
        # If using step method, call it directly with original hidden_states
        if use_step:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch_size)
            
            # Reshape hidden_states for step: (batch * seqlen, hidden_size) -> (batch, seqlen, hidden_size)
            # Step expects (batch, 1, hidden_size) for single token
            # Reference asserts seqlen == 1, so we reshape accordingly
            hidden_states_step = ops.reshape(
                x,
                shape=[batch_size, seqlen, self.hidden_size],
            )
            
            # For seqlen=1, reshape to (batch, 1, hidden_size)
            # For seqlen>1, we'd need to loop, but typically step is called with seqlen=1
            # Reshape to (batch, 1, hidden_size) - step method will handle both shapes
            if hasattr(seqlen, '__int__'):
                try:
                    if int(seqlen) == 1:
                        hidden_states_step = ops.reshape(hidden_states_step, shape=[batch_size, 1, self.hidden_size])
                except (TypeError, ValueError):
                    # Symbolic dimension, assume seqlen=1 and reshape
                    hidden_states_step = ops.reshape(hidden_states_step, shape=[batch_size, 1, self.hidden_size])
            else:
                # Assume seqlen=1 for autoregressive mode
                hidden_states_step = ops.reshape(hidden_states_step, shape=[batch_size, 1, self.hidden_size])
            
            # Call step method
            out_step, conv_state, ssm_state = self.step(
                hidden_states_step,
                conv_state,
                ssm_state,
            )
            
            # Update cache
            if "key_value_memory_dict" in inference_params:
                inference_params["key_value_memory_dict"][self.layer_idx] = (conv_state, ssm_state)
            
            # Reshape output: (batch, 1, hidden_size) -> (batch, hidden_size)
            ss_output = ops.reshape(out_step, shape=[batch_size, self.hidden_size])
            return ss_output
        
        # Step 2: Input projection and gating (for prefill mode)
        # Infer batch and seqlen for reshaping
        # For now, simplify and use batch_size=1 to avoid ragged tensor complexity
        # TODO: Implement proper ragged tensor support
        batch_size = 1
        seqlen = x.shape[0]
        
        # Project input - matches reference: "We do matmul and transpose BLH -> HBL at the same time"
        # Reference does: rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        # In MAX, we work with flattened tensors, so we project then reshape
        xz_flat = self.in_proj(x)  # (batch * seqlen, intermediate_size * 2)
        
        # Reshape to (batch, seqlen, 2 * intermediate_size) first, then permute to (batch, 2 * intermediate_size, seqlen)
        # This is critical: reshape alone doesn't reorder elements, we need reshape + permute
        xz_temp = ops.reshape(
            xz_flat,
            shape=[batch_size, seqlen, self.intermediate_size * 2],
        )
        # Permute to (batch, 2 * intermediate_size, seqlen) to match reference format
        xz = ops.permute(xz_temp, [0, 2, 1])
        
        # Check if we should use fast path (mamba_inner_fn)
        # Reference: use_fast_path and causal_conv1d_fn is not None and inference_params is None
        use_fast_path = (
            self.use_fast_path
            and causal_conv1d_fn is not None
            and inference_params is None
        )
        
        if use_fast_path:
            # Fast path: use mamba_inner_fn
            # Prepare A and D
            A_log_cast = self.A_log.cast(self.dtype)
            A = ops.negate(ops.exp(A_log_cast))  # (intermediate_size, d_state)
            D = self.D.cast(self.dtype)
            
            # Get delta_bias
            if hasattr(self.dt_proj, 'bias') and self.dt_proj.bias is not None:
                delta_bias = self.dt_proj.bias.cast(self.dtype)
            else:
                delta_bias = None
            
            # Call mamba_inner_fn
            # xz: (batch, 2 * intermediate_size, seqlen)
            # Returns: (batch, seqlen, hidden_size)
            out = mamba_inner_fn(
                xz=xz,
                conv1d_weight=self.conv1d_weight,
                conv1d_bias=self.conv1d_bias,
                x_proj_weight=self.x_proj.weight,
                delta_proj_weight=self.dt_proj.weight,
                out_proj_weight=self.out_proj.weight,
                out_proj_bias=self.out_proj.bias if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None else None,
                A=A,
                B=None,  # Will be computed from x_proj output
                C=None,  # Will be computed from x_proj output
                D=D,
                delta_bias=delta_bias,
                delta_softplus=self.delta_softplus,
            )
            
            # Reshape output back to flattened format: (batch, seqlen, hidden_size) -> (batch * seqlen, hidden_size)
            out_flat = ops.reshape(
                out,
                shape=[batch_size * seqlen, self.hidden_size],
            )
            return out_flat
        else:
            # Slow path: manual computation (matches reference else branch)
            # Split into x and z (gate and up in reference terminology)
            x, z = ops.split(
                xz,
                [self.intermediate_size, self.intermediate_size],
                axis=1,
            )

            x_conv_input = x  # Already in (batch, intermediate_size, seqlen)
            z_conv = z  # Already in (batch, intermediate_size, seqlen)

            # Apply conv1d to x
            x_conv = causal_conv1d_fn(
                x_conv_input,
                self.conv1d_weight,
                bias=self.conv1d_bias,
                algorithm="optimized",
                activation="silu",
            )  # (batch, intermediate_size, seqlen)

            # Permute and reshape for linear projections
            # (batch, intermediate_size, seqlen) -> (batch, seqlen, intermediate_size) -> (batch * seqlen, intermediate_size)
            x_conv_permuted = ops.permute(x_conv, [0, 2, 1])  # (batch, seqlen, intermediate_size)
            x_flat = ops.reshape(
                x_conv_permuted,
                shape=[batch_size * seqlen, self.intermediate_size],
            )

            # Project to get B, C, delta
            # Reference: x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            x_proj = self.x_proj(x_flat)  # (batch * seqlen, dt_rank + 2 * d_state)

            # Split x_proj into delta, B, C
            # Reference: dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
            dt, B, C = ops.split(
                x_proj,
                [self.dt_rank, self.d_state, self.d_state],
                axis=-1,
            )

            # Project delta
            # Reference: dt = self.dt_proj.weight @ dt.t()
            dt_proj = self.dt_proj(dt)  # (batch * seqlen, intermediate_size)

            # Reshape for selective scan forward
            # Reference: dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            # (batch * seqlen, intermediate_size) -> (batch, seqlen, intermediate_size) -> (batch, intermediate_size, seqlen)
            dt_reshaped = ops.reshape(dt_proj, shape=[batch_size, seqlen, self.intermediate_size])
            delta = ops.permute(dt_reshaped, [0, 2, 1])  # (batch, intermediate_size, seqlen)
            
            # u is the conv output in (batch, intermediate_size, seqlen) format
            u = x_conv  # Already in correct shape
            
            # Reshape B and C: (batch * seqlen, d_state) -> (batch, 1, d_state, seqlen)
            # Reference: B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen)
            # For selective_scan_fn: B should be (batch, n_groups, d_state, seqlen) with n_groups=1
            # Step 1: (batch * seqlen, d_state) -> (batch, seqlen, d_state)
            B_temp = ops.reshape(B, shape=[batch_size, seqlen, self.d_state])
            # Step 2: (batch, seqlen, d_state) -> (batch, d_state, seqlen)
            B_temp = ops.permute(B_temp, [0, 2, 1])
            # Step 3: Add n_groups dimension: (batch, d_state, seqlen) -> (batch, 1, d_state, seqlen)
            B_reshaped = ops.reshape(B_temp, shape=[batch_size, 1, self.d_state, seqlen])
            
            # Same for C
            C_temp = ops.reshape(C, shape=[batch_size, seqlen, self.d_state])
            C_temp = ops.permute(C_temp, [0, 2, 1])
            C_reshaped = ops.reshape(C_temp, shape=[batch_size, 1, self.d_state, seqlen])

            # Prepare A and D
            A_log_cast = self.A_log.cast(self.dtype)
            A = ops.negate(ops.exp(A_log_cast))  # (intermediate_size, d_state)
            D = self.D.cast(self.dtype)

            # delta_bias - NOTE: Since self.dt_proj is a Linear layer that already includes bias,
            # we should NOT pass delta_bias separately to the kernel (which would double-apply it).
            # Pass a zero tensor to avoid affecting the computation.
            delta_bias = ops.constant(0.0, dtype=self.dtype, device=u.device)
            delta_bias = ops.broadcast_to(delta_bias, shape=[self.intermediate_size])  # All zeros

            # selective_scan_fwd outputs
            chunk_size = 2048
            num_chunks = (seqlen + Dim(chunk_size) - Dim(1)) // Dim(chunk_size)

            output_type = TensorType(
                dtype=self.dtype,
                shape=[batch_size, self.intermediate_size, seqlen],
                device=u.device,
            )
            x_checkpoint_type = TensorType(
                dtype=self.dtype,
                shape=[batch_size, self.intermediate_size, num_chunks, 2 * self.d_state],
                device=u.device,
            )
            out_z_type = TensorType(
                dtype=self.dtype,
                shape=[batch_size, self.intermediate_size, seqlen],
                device=u.device,
            )

            results = ops.custom(
                "selective_scan_fwd",
                device=u.device,
                values=[u, delta, A, B_reshaped, C_reshaped, D, z_conv, delta_bias],
                out_types=[output_type, x_checkpoint_type, out_z_type],
                parameters={"delta_softplus": self.delta_softplus},
            )

            # Use out_z (results[2]) which is the gated output: output * silu(z)
            # results[0] is the ungated output, results[1] is the checkpoint tensor
            ss_output = results[2].tensor  # (batch, intermediate_size, seqlen)

            # Permute to (batch, seqlen, intermediate_size) then flatten for output projection
            # Reference: y = rearrange(y, "b d l -> b l d")
            ss_output_permuted = ops.permute(ss_output, [0, 2, 1])  # (batch, seqlen, intermediate_size)
            ss_output_flat = ops.reshape(
                ss_output_permuted,
                shape=[batch_size * seqlen, self.intermediate_size],
            )
            return self.out_proj(ss_output_flat)
    
    def step(
        self,
        hidden_states: TensorValue,
        conv_state: TensorValue,
        ssm_state: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Single step forward pass for autoregressive generation.
        
        Matches reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            hidden_states: Input hidden states of shape (batch, 1, hidden_size) or (batch, hidden_size).
            conv_state: Convolution state buffer (batch, intermediate_size, width-1).
            ssm_state: SSM state buffer (batch, intermediate_size, d_state).
            
        Returns:
            Tuple of (output, conv_state, ssm_state) where:
            - output: Output tensor of shape (batch, 1, hidden_size).
            - conv_state: Updated convolution state.
            - ssm_state: Updated SSM state.
        """
        # Reference: assert hidden_states.shape[1] == 1
        # Handle both (batch, 1, hidden_size) and (batch, hidden_size) shapes
        if len(hidden_states.shape.static_dims) == 3:
            # (batch, 1, hidden_size) -> (batch, hidden_size)
            batch_size = hidden_states.shape[0]
            hidden_states = ops.reshape(hidden_states, shape=[batch_size, self.hidden_size])
        else:
            # (batch, hidden_size)
            batch_size = hidden_states.shape[0]
        
        # Project input: xz = self.in_proj(hidden_states.squeeze(1))
        xz = self.in_proj(hidden_states)  # (batch, intermediate_size * 2)
        
        # Split into x and z
        x, z = ops.split(
            xz,
            [self.intermediate_size, self.intermediate_size],
            axis=-1,
        )  # (batch, intermediate_size) each
        
        # Conv step
        # Reshape x for conv1d: (batch, intermediate_size) -> (batch, intermediate_size, 1)
        x_conv = ops.reshape(x, shape=[batch_size, self.intermediate_size, 1])
        
        # Use causal_conv1d_update_fn
        x_conv = causal_conv1d_update_fn(
            x_conv,
            conv_state,
            self.conv1d_weight,
            bias=self.conv1d_bias,
            activation="silu",
        )  # (batch, intermediate_size, 1)
        
        # Reshape back: (batch, intermediate_size, 1) -> (batch, intermediate_size)
        x = ops.reshape(x_conv, shape=[batch_size, self.intermediate_size])
        
        # Project to get dt, B, C
        # Reference: x_db = self.x_proj(x)  # (B, dt_rank + 2*d_state)
        x_db = self.x_proj(x)  # (batch, dt_rank + 2 * d_state)
        
        # Split into dt, B, C
        # Reference: dt, B, C = torch.split(x_db, [dt_rank, d_state, d_state], dim=-1)
        dt, B, C = ops.split(
            x_db,
            [self.dt_rank, self.d_state, self.d_state],
            axis=-1,
        )
        
        # Project delta (don't add dt_bias here, kernel handles it)
        # Reference: dt = F.linear(dt, self.dt_proj.weight)
        dt = self.dt_proj(dt)  # (batch, intermediate_size)
        
        # Reshape B and C for update kernel: (batch, n_groups, d_state)
        # n_groups=1 for standard Mamba
        B_grouped = ops.reshape(B, shape=[batch_size, 1, self.d_state])
        C_grouped = ops.reshape(C, shape=[batch_size, 1, self.d_state])
        
        # Prepare A and D (same as forward)
        A_log_cast = self.A_log.cast(self.dtype)
        A = ops.negate(ops.exp(A_log_cast))  # (intermediate_size, d_state)
        D = self.D.cast(self.dtype)

        # Get delta_bias
        if hasattr(self.dt_proj, 'bias') and self.dt_proj.bias is not None:
            delta_bias = self.dt_proj.bias.cast(self.dtype)
        else:
            delta_bias = ops.constant(0.0, dtype=self.dtype, device=hidden_states.device)
            delta_bias = ops.broadcast_to(delta_bias, shape=[self.intermediate_size])
        
        # SSM step: call selective_scan_update
        state_out_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size, self.d_state],
            device=hidden_states.device,
        )
        output_type = TensorType(
            dtype=self.dtype,
            shape=[batch_size, self.intermediate_size],
            device=hidden_states.device,
        )
        
        results = ops.custom(
            "selective_scan_update",
            device=hidden_states.device,
            values=[ssm_state, x, dt, A, B_grouped, C_grouped, D, z, delta_bias],
            out_types=[state_out_type, output_type],
            parameters={"delta_softplus": self.delta_softplus},
        )
        
        # Update states
        ssm_state = results[0].tensor  # Updated SSM state
        y = results[1].tensor  # (batch, intermediate_size)
        
        # Output projection
        out = self.out_proj(y)  # (batch, hidden_size)
        
        # Reshape to (batch, 1, hidden_size) for consistency
        out = ops.reshape(out, shape=[batch_size, 1, self.hidden_size])
        
        return out, conv_state, ssm_state
    
    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: DType | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Allocate inference cache buffers for autoregressive generation.
        
        Matches reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Reference shapes:
        - conv_state: (batch_size, d_model * expand, d_conv)
        - ssm_state: (batch_size, d_model * expand, d_state)
        
        Args:
            batch_size: Batch size for the cache.
            max_seqlen: Maximum sequence length (not used for Mamba, but kept for compatibility).
            dtype: Data type for cache (uses conv1d weight dtype if None).
            
        Returns:
            Tuple of (conv_state, ssm_state) buffers.
        """
        conv_dtype = self.conv1d_weight.dtype if dtype is None else dtype
        conv_state = ops.constant(0.0, dtype=conv_dtype, device=self.device)
        conv_state = ops.broadcast_to(
            conv_state,
            shape=[batch_size, self.intermediate_size, self.conv_width],
        )
        
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = ops.constant(0.0, dtype=ssm_dtype, device=self.device)
        ssm_state = ops.broadcast_to(
            ssm_state,
            shape=[batch_size, self.intermediate_size, self.d_state],
        )
        
        return conv_state, ssm_state
    
    def _get_states_from_cache(
        self,
        inference_params: dict,
        batch_size: int,
        initialize_states: bool = False,
    ) -> tuple[TensorValue, TensorValue]:
        """Get or create state buffers from inference cache.
        
        Matches reference implementation:
        https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        
        Args:
            inference_params: Inference parameters dict containing key_value_memory_dict.
            batch_size: Batch size for the states.
            initialize_states: Whether to initialize states to zero.
            
        Returns:
            Tuple of (conv_state, ssm_state) buffers.
        """
        assert self.layer_idx is not None, "layer_idx must be set for inference caching"
        
        if "key_value_memory_dict" not in inference_params:
            inference_params["key_value_memory_dict"] = {}
        
        cache_dict = inference_params["key_value_memory_dict"]
        
        if self.layer_idx not in cache_dict:
            # Allocate new states
            conv_state, ssm_state = self.allocate_inference_cache(
                batch_size,
                max_seqlen=1,  # Not used but required by signature
            )
            cache_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = cache_dict[self.layer_idx]
            if initialize_states:
                # Zero out states
                conv_state = ops.constant(0.0, dtype=conv_state.dtype, device=self.device)
                conv_state = ops.broadcast_to(conv_state, shape=conv_state.shape)
                ssm_state = ops.constant(0.0, dtype=ssm_state.dtype, device=self.device)
                ssm_state = ops.broadcast_to(ssm_state, shape=ssm_state.shape)
                cache_dict[self.layer_idx] = (conv_state, ssm_state)
        
        return conv_state, ssm_state


class Block(Module):
    """Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.
    
    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
    """

    def __init__(
        self,
        dim: int,
        mixer: Module,
        mlp: Layer | None = None,
        norm: Layer | None = None,
        norm2: Layer | None = None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ) -> None:
        """Initialize a Mamba block.
        
        Args:
            dim: Hidden dimension size.
            mixer: The mixer class (SSM layer, replaces attention in transformers).
            mlp: Optional MLP layer. If None, MLP is skipped.
            norm: Normalization layer (typically RMSNorm or LayerNorm) for the mixer path.
            norm2: Optional second normalization layer for the MLP path.
            fused_add_norm: Whether to use fused add-norm operations via layer_norm_fn.
            residual_in_fp32: Whether to keep residual in float32 precision.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm
        self.mixer = mixer
        if mlp is not None:
            self.norm2 = norm2
            self.mlp = mlp
        else:
            self.mlp = None
            self.norm2 = None
        
        if self.fused_add_norm:
            # Verify that layer_norm_fn is available
            if layer_norm_fn is None:
                raise ValueError("layer_norm_fn import fails")
            # Only LayerNorm and RMSNorm are supported for fused_add_norm
            # We use duck typing (attribute checks) since Protocol classes can't be used with isinstance
            def _is_valid_norm(norm: Layer | None) -> bool:
                if norm is None:
                    return True
                # Check for required attributes that all norm layers should have
                return hasattr(norm, 'weight') and hasattr(norm, 'eps')
            
            if not _is_valid_norm(self.norm):
                raise ValueError(
                    "Only LayerNorm and FusedRMSNorm are supported for fused_add_norm, "
                    f"got {type(self.norm)}"
                )
            if not _is_valid_norm(self.norm2):
                raise ValueError(
                    "Only LayerNorm and FusedRMSNorm are supported for fused_add_norm, "
                    f"got {type(self.norm2)}"
                )

    def __call__(
        self,
        hidden_states: TensorValue,
        residual: Optional[TensorValue] = None,
        input_row_offsets: TensorValue | None = None,
        inference_params: Optional[dict] = None,
        **mixer_kwargs,
    ) -> tuple[TensorValue, TensorValue]:
        """Forward pass through the block.
        
        Args:
            hidden_states: The sequence to the block (required).
            residual: Optional residual tensor. If None, uses hidden_states as residual.
            input_row_offsets: Row offsets for ragged tensor processing.
            inference_params: Optional inference parameters for caching.
            **mixer_kwargs: Additional keyword arguments for the mixer.
            
        Returns:
            Tuple of (hidden_states, residual) where:
            - hidden_states: Output hidden states after mixer (and optionally MLP)
            - residual: Updated residual tensor
        """
        if not self.fused_add_norm:
            # Non-fused path: add residual first, then normalize
            if self.norm is not None:
                if residual is not None:
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for addition") + residual
                else:
                    residual = hidden_states
                # Cast to norm dtype for normalization
                norm_dtype = self.norm.weight.dtype if hasattr(self.norm, 'weight') else residual.dtype
                hidden_states = self.norm(residual.cast(norm_dtype))
                if self.residual_in_fp32:
                    residual = residual.cast(DType.float32)
            else:
                # No norm, just use hidden_states as residual
                residual = (hidden_states + residual) if residual is not None else hidden_states
        else:
            # Fused path: use layer_norm_fn for fused add-norm operation
            if self.norm is not None:
                # Get weight and bias, handling device placement
                norm_weight = self.norm.weight.cast(hidden_states.dtype)
                if hidden_states.device:
                    norm_weight = norm_weight.to(hidden_states.device)
                
                # Check if this is an RMSNorm by class name (avoid isinstance due to Protocol issues)
                norm_type_name = type(self.norm).__name__
                is_rms_norm = 'RMSNorm' in norm_type_name
                
                # Get bias (can be None for RMSNorm)
                norm_bias = None
                if not is_rms_norm and hasattr(self.norm, 'bias') and self.norm.bias is not None:
                    norm_bias = self.norm.bias.cast(hidden_states.dtype)
                    if hidden_states.device:
                        norm_bias = norm_bias.to(hidden_states.device)
                
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    norm_weight,
                    bias=norm_bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                    is_rms_norm=is_rms_norm,
                )
            else:
                # No norm, just use hidden_states as residual
                residual = (hidden_states + residual) if residual is not None else hidden_states
        
        # Apply mixer (SSM)
        if input_row_offsets is not None:
            hidden_states = self.mixer(
                hidden_states,
                input_row_offsets=input_row_offsets,
                inference_params=inference_params,
                **mixer_kwargs,
            )
        else:
            hidden_states = self.mixer(
                hidden_states,
                inference_params=inference_params,
                **mixer_kwargs,
            )

        # Apply MLP if present
        if self.mlp is not None:
            if not self.fused_add_norm:
                # Non-fused path: add residual first, then normalize
                if self.norm2 is not None:
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for MLP residual addition") + residual
                    # Cast to norm dtype for normalization
                    norm2_dtype = self.norm2.weight.dtype if hasattr(self.norm2, 'weight') else residual.dtype
                    hidden_states = self.norm2(residual.to(dtype=norm2_dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(dtype=DType.float32)
                else:
                    # No norm2, just add residual
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for MLP residual addition (no norm2)") + residual
            else:
                # Fused path: use layer_norm_fn for fused add-norm operation
                if self.norm2 is not None:
                    # Get weight and bias, handling device placement
                    norm2_weight = self.norm2.weight.cast(hidden_states.dtype)
                    if hidden_states.device:
                        norm2_weight = norm2_weight.to(hidden_states.device)
                    
                    # Check if this is an RMSNorm by class name (avoid isinstance due to Protocol issues)
                    norm2_type_name = type(self.norm2).__name__
                    is_rms_norm2 = 'RMSNorm' in norm2_type_name
                    
                    # Get bias (can be None for RMSNorm)
                    norm2_bias = None
                    if not is_rms_norm2 and hasattr(self.norm2, 'bias') and self.norm2.bias is not None:
                        norm2_bias = self.norm2.bias.cast(hidden_states.dtype)
                        if hidden_states.device:
                            norm2_bias = norm2_bias.to(hidden_states.device)
                    
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        norm2_weight,
                        bias=norm2_bias,
                        residual=residual,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        eps=self.norm2.eps,
                        is_rms_norm=is_rms_norm2,
                    )
                else:
                    # No norm2, just add residual
                    # Use rebind to assert shapes are equivalent at runtime
                    residual = hidden_states.rebind(residual.shape, message="Asserting hidden_states and residual have equivalent shapes for MLP residual addition (fused, no norm2)") + residual
            
            # Apply MLP
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

