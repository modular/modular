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
"""SSM state cache for Mamba models.

This module provides cache infrastructure for Mamba's selective scan state (SSM state)
analogous to the KV cache used in transformer attention.

Unlike KV cache which stores key-value pairs for attention, SSM state cache stores:
- conv_state: Convolution state of shape (batch, intermediate_size, conv_kernel)
- ssm_state: Selective scan state of shape (batch, intermediate_size, d_state)
"""

from __future__ import annotations

from dataclasses import dataclass

from max.driver import Tensor
from max.dtype import DType
from max.graph import BufferType, BufferValue, DeviceRef, TensorType, TensorValue


@dataclass
class SSMStateInputSymbols:
    """Input type symbols for SSM state cache in graph building.
    
    These define the tensor types that will be passed as graph inputs
    for SSM state caching during autoregressive generation.
    """
    
    # Conv state: (num_layers, batch, intermediate_size, conv_kernel)
    conv_state: BufferType
    # SSM state: (num_layers, batch, intermediate_size, d_state)  
    ssm_state: BufferType
    # Sequence offset to distinguish prefill (0) from step (>0)
    seqlen_offset: TensorType
    
    def __iter__(self):
        """Iterate through input types."""
        yield self.conv_state
        yield self.ssm_state
        yield self.seqlen_offset


@dataclass
class SSMStateValues:
    """SSM state values passed through the graph during execution.
    
    These hold the actual buffer/tensor values during graph execution.
    """
    
    conv_state: BufferValue
    ssm_state: BufferValue
    seqlen_offset: TensorValue
    
    def __iter__(self):
        """Iterate through values."""
        yield self.conv_state
        yield self.ssm_state
        yield self.seqlen_offset


@dataclass
class SSMStateCacheInputs:
    """Runtime inputs for SSM state cache (Tensor form).
    
    These are the actual Tensors passed to model.execute() at runtime.
    """
    
    conv_state: Tensor
    ssm_state: Tensor
    seqlen_offset: Tensor
    
    def __iter__(self):
        """Iterate through tensors."""
        yield self.conv_state
        yield self.ssm_state
        yield self.seqlen_offset
    
    def __len__(self) -> int:
        return 3


@dataclass 
class SSMStateCacheParams:
    """Parameters for SSM state cache configuration.
    
    Similar to KVCacheParams but for Mamba's SSM state.
    """
    
    dtype: DType
    """Data type for SSM state tensors."""
    
    num_layers: int
    """Number of Mamba layers."""
    
    intermediate_size: int
    """Intermediate dimension (d_inner) of the SSM."""
    
    d_state: int
    """State dimension of the SSM."""
    
    conv_kernel: int
    """Convolution kernel size (d_conv)."""
    
    device: DeviceRef
    """Device for the state tensors."""
    
    def get_input_symbols(self) -> SSMStateInputSymbols:
        """Get graph input type symbols for SSM state cache."""
        # Conv state: (num_layers, batch, intermediate_size, conv_kernel)
        # Using symbolic batch dimension
        conv_state_type = BufferType(
            self.dtype,
            shape=[self.num_layers, "batch", self.intermediate_size, self.conv_kernel],
            device=self.device,
        )
        
        # SSM state: (num_layers, batch, intermediate_size, d_state)
        ssm_state_type = BufferType(
            self.dtype,
            shape=[self.num_layers, "batch", self.intermediate_size, self.d_state],
            device=self.device,
        )
        
        # Seqlen offset: scalar
        seqlen_offset_type = TensorType(
            DType.int64,
            shape=[1],
            device=DeviceRef.CPU(),
        )
        
        return SSMStateInputSymbols(
            conv_state=conv_state_type,
            ssm_state=ssm_state_type,
            seqlen_offset=seqlen_offset_type,
        )
    
    def allocate_cache(self, batch_size: int) -> SSMStateCacheInputs:
        """Allocate SSM state cache tensors for a given batch size.
        
        Args:
            batch_size: Number of sequences in the batch.
            
        Returns:
            SSMStateCacheInputs with zero-initialized state tensors.
        """
        import numpy as np
        
        # Initialize conv_state to zeros
        conv_state_np = np.zeros(
            (self.num_layers, batch_size, self.intermediate_size, self.conv_kernel),
            dtype=self.dtype.to_numpy(),
        )
        conv_state = Tensor.from_numpy(conv_state_np).to(self.device)
        
        # Initialize ssm_state to zeros
        ssm_state_np = np.zeros(
            (self.num_layers, batch_size, self.intermediate_size, self.d_state),
            dtype=self.dtype.to_numpy(),
        )
        ssm_state = Tensor.from_numpy(ssm_state_np).to(self.device)
        
        # Initialize seqlen_offset to 0 (prefill mode)
        seqlen_offset = Tensor.from_numpy(np.array([0], dtype=np.int64))
        
        return SSMStateCacheInputs(
            conv_state=conv_state,
            ssm_state=ssm_state,
            seqlen_offset=seqlen_offset,
        )


def create_ssm_state_params(
    dtype: DType,
    num_layers: int,
    intermediate_size: int,
    d_state: int,
    conv_kernel: int,
    device: DeviceRef,
) -> SSMStateCacheParams:
    """Create SSM state cache parameters.
    
    Args:
        dtype: Data type for state tensors.
        num_layers: Number of Mamba layers.
        intermediate_size: Intermediate dimension of the SSM.
        d_state: State dimension of the SSM.
        conv_kernel: Convolution kernel size.
        device: Device for state tensors.
        
    Returns:
        SSMStateCacheParams instance.
    """
    return SSMStateCacheParams(
        dtype=dtype,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        d_state=d_state,
        conv_kernel=conv_kernel,
        device=device,
    )
