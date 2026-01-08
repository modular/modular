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
"""Tests for selective scan operations in max.nn."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.dtype import DType
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, ops
from max.nn.conv import causal_conv1d_fn
from max.nn.selective_scan import (
    mamba_inner_fn,
    mamba_inner_fn_simplified,
    mamba_inner_ref,
    selective_scan_fn,
    selective_state_update_fn,
)


def _get_mamba_kernel_api_path() -> Path | None:
    """Get path to MambaKernelAPI.mojopkg for custom extensions.
    
    Looks for MambaKernelAPI.mojopkg in MODULAR_MOJO_MAX_IMPORT_PATH
    environment variable set by Bazel when mojo_deps are specified.
    """
    import_path_env = os.environ.get("MODULAR_MOJO_MAX_IMPORT_PATH", "")
    if not import_path_env:
        return None
    
    for entry in import_path_env.split(","):
        if not entry.strip():
            continue
        
        entry_path = Path(entry.strip())
        if not entry_path.is_absolute():
            resolved = Path.cwd() / entry_path
            if not resolved.exists():
                resolved = entry_path
            entry_path = resolved
        
        if not entry_path.exists():
            continue
        
        # If it's already a .mojopkg file
        if entry_path.suffix == ".mojopkg":
            if "MambaKernelAPI" in entry_path.name:
                return entry_path.resolve()
            continue
        
        # If it's a directory, search for MambaKernelAPI.mojopkg
        if entry_path.is_dir():
            for mojopkg in entry_path.rglob("*.mojopkg"):
                if "MambaKernelAPI" in mojopkg.name:
                    return mojopkg.resolve()
    
    return None




@pytest.fixture(scope="function")
def session():
    """Create an inference session for testing.
    
    Using function scope with explicit cleanup to ensure each test gets a fresh session,
    preventing GPU state corruption between tests.
    """
    device = CPU() if accelerator_count() == 0 else Accelerator()
    sess = InferenceSession(devices=[device])
    yield sess
    # Explicit cleanup to prevent GPU state corruption
    del sess
    gc.collect()


@pytest.fixture
def device() -> DeviceRef:
    """Get device for testing."""
    return DeviceRef.CPU() if accelerator_count() == 0 else DeviceRef.GPU(0)


@pytest.fixture
def mamba_kernel_path() -> list[Path]:
    """Get path to MambaKernelAPI for custom extensions."""
    path = _get_mamba_kernel_api_path()
    return [path] if path else []


def create_test_data(
    batch: int = 2,
    dim: int = 4,
    seqlen: int = 8,
    dstate: int = 2,
    n_groups: int = 1,
    dtype: DType = DType.float32,
    ) -> tuple[np.ndarray[Any, Any], ...]:
    """Create test data for selective scan.
    
    Returns:
        Tuple of (u, delta, A, B, C, D, z, delta_bias) as numpy arrays.
    """
    np_dtype = np.float32 if dtype == DType.float32 else np.float16
    
    # u: (batch, dim, seqlen)
    u = np.random.randn(batch, dim, seqlen).astype(np_dtype)
    
    # delta: (batch, dim, seqlen) - positive values
    delta = np.random.uniform(0.1, 1.0, (batch, dim, seqlen)).astype(np_dtype)
    
    # A: (dim, dstate) - typically negative for stability
    A = np.random.uniform(-2.0, -0.1, (dim, dstate)).astype(np_dtype)
    
    # B: (batch, n_groups, dstate, seqlen)
    B = np.random.randn(batch, n_groups, dstate, seqlen).astype(np_dtype)
    
    # C: (batch, n_groups, dstate, seqlen)
    C = np.random.randn(batch, n_groups, dstate, seqlen).astype(np_dtype)
    
    # D: (dim,) - skip connection
    D = np.random.randn(dim).astype(np_dtype)
    
    # z: (batch, dim, seqlen) - gate tensor
    z = np.random.randn(batch, dim, seqlen).astype(np_dtype)
    
    # delta_bias: (dim,)
    delta_bias = np.random.randn(dim).astype(np_dtype)
    
    return u, delta, A, B, C, D, z, delta_bias


class TestSelectiveScanFn:
    """Tests for selective_scan_fn."""
    
    def test_selective_scan_minimal_gpu(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Minimal GPU test - no optional params, simplest possible case."""
        batch, dim, seqlen, dstate, n_groups = 1, 2, 4, 2, 1

        # Create simple test data
        np.random.seed(42)
        u = np.random.randn(batch, dim, seqlen).astype(np.float32)
        delta = np.random.randn(batch, dim, seqlen).astype(np.float32)
        A = -np.random.rand(dim, dstate).astype(np.float32)  # Negative for stability
        B = np.random.randn(batch, n_groups, dstate, seqlen).astype(np.float32)
        C = np.random.randn(batch, n_groups, dstate, seqlen).astype(np.float32)

        dtype = DType.float32
        # Only required inputs - no D, z, or delta_bias
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
        ]

        graph = Graph(
            "test_selective_scan_minimal",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor

            # Call with minimal params - no D, z, or delta_bias
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=None,  # No skip connection
                z=None,  # No gating
                delta_bias=None,  # No bias
                delta_softplus=False,
            )
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = session.load(graph)

        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
        ]

        results = compiled_model.execute(*inputs)

        assert results[0].shape == (batch, dim, seqlen)
    
    def test_selective_scan_cpu_only(
        self, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective_scan_fn on CPU only to isolate GPU issues."""
        # Force CPU device
        cpu_session = InferenceSession(devices=[CPU()])
        cpu_device = DeviceRef.CPU()

        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1

        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], cpu_device),  # u
            TensorType(dtype, [batch, dim, seqlen], cpu_device),  # delta
            TensorType(dtype, [dim, dstate], cpu_device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], cpu_device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], cpu_device),  # C
            TensorType(dtype, [dim], cpu_device),  # D
        ]

        graph = Graph(
            "test_selective_scan_cpu_only",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor

            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_softplus=False,
            )
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = cpu_session.load(graph)

        inputs = [
            Tensor.from_numpy(u).to(cpu_session.devices[0]),
            Tensor.from_numpy(delta).to(cpu_session.devices[0]),
            Tensor.from_numpy(A).to(cpu_session.devices[0]),
            Tensor.from_numpy(B).to(cpu_session.devices[0]),
            Tensor.from_numpy(C).to(cpu_session.devices[0]),
            Tensor.from_numpy(D).to(cpu_session.devices[0]),
        ]

        results = compiled_model.execute(*inputs)

        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)
    
    def test_selective_scan_minimal_cpu(
        self, mamba_kernel_path: list[Path]
    ) -> None:
        """Test minimal selective_scan_fn on CPU - no D, z, or delta_bias."""
        # Force CPU device
        cpu_session = InferenceSession(devices=[CPU()])
        cpu_device = DeviceRef.CPU()
        
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        
        
        np.random.seed(42)
        u = np.random.randn(batch, dim, seqlen).astype(np.float32)
        delta = np.random.uniform(0.1, 1.0, (batch, dim, seqlen)).astype(np.float32)
        A = np.random.uniform(-2.0, -0.1, (dim, dstate)).astype(np.float32)
        B = np.random.randn(batch, n_groups, dstate, seqlen).astype(np.float32)
        C = np.random.randn(batch, n_groups, dstate, seqlen).astype(np.float32)
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], cpu_device),  # u
            TensorType(dtype, [batch, dim, seqlen], cpu_device),  # delta
            TensorType(dtype, [dim, dstate], cpu_device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], cpu_device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], cpu_device),  # C
        ]
        
        graph = Graph(
            "test_selective_scan_minimal_cpu",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            
            # Call with minimal params - no D, z, or delta_bias
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=None,  # No skip connection
                z=None,  # No gating
                delta_bias=None,  # No bias
                delta_softplus=False,
            )
            assert not isinstance(output, tuple)
            graph.output(output)
        
        compiled_model = cpu_session.load(graph)
        
        inputs = [
            Tensor.from_numpy(u).to(cpu_session.devices[0]),
            Tensor.from_numpy(delta).to(cpu_session.devices[0]),
            Tensor.from_numpy(A).to(cpu_session.devices[0]),
            Tensor.from_numpy(B).to(cpu_session.devices[0]),
            Tensor.from_numpy(C).to(cpu_session.devices[0]),
        ]
        
        results = compiled_model.execute(*inputs)
        
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)
    
    def test_selective_scan_debug(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Debug test to print tensor information."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        
        
        # Create test data
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )
        
        
        # Expected strides for contiguous tensors (in bytes, float32 = 4 bytes)
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim], device),  # D
        ]
        
        graph = Graph(
            "test_selective_scan_debug",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            
            
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_softplus=False,
            )
            assert not isinstance(output, tuple)
            graph.output(output)
        
        compiled_model = session.load(graph)
        
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        
        results = compiled_model.execute(*inputs)
        
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)
    
    def test_selective_scan_basic(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test basic selective scan forward pass."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim], device),  # D
        ]
        
        graph = Graph(
            "test_selective_scan_basic",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)
    
    def test_selective_scan_with_z(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective scan with gate tensor z."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim], device),  # D
            TensorType(dtype, [batch, dim, seqlen], device),  # z
        ]
        
        graph = Graph(
            "test_selective_scan_with_z",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            z_val = graph.inputs[6].tensor
            
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                z=z_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
            Tensor.from_numpy(z).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)
    
    def test_selective_scan_with_delta_bias(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective scan with delta_bias."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim], device),  # D
            TensorType(dtype, [dim], device),  # delta_bias
        ]
        
        graph = Graph(
            "test_selective_scan_with_delta_bias",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            delta_bias_val = graph.inputs[6].tensor
            
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_bias=delta_bias_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
            Tensor.from_numpy(delta_bias).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)
    
    def test_selective_scan_return_last_state(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective scan with return_last_state=True."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim], device),  # D
        ]
        
        graph = Graph(
            "test_selective_scan_return_last_state",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            
            result = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_softplus=False,
                return_last_state=True,
            )
            # result is a tuple (output, last_state) when return_last_state=True
            assert isinstance(result, tuple)
            output, last_state = result
            graph.output(output, last_state)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shapes
        assert len(results) == 2
        output_tensor = results[0]
        last_state_tensor = results[1]
        assert output_tensor.shape == (batch, dim, seqlen)
        assert last_state_tensor.shape == (batch, dim, dstate)
    
    def test_selective_scan_delta_softplus(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective scan with delta_softplus=True."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )
        
        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim], device),  # D
        ]
        
        graph = Graph(
            "test_selective_scan_delta_softplus",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_softplus=True,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)


class TestSelectiveStateUpdateFn:
    """Tests for selective_state_update_fn."""
    
    def test_selective_state_update_basic(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test basic selective state update."""
        batch, dim, dstate, n_groups = 2, 4, 2, 1
        dtype = DType.float32
        np_dtype = np.float32
        
        # state: (batch, dim, dstate)
        state = np.random.randn(batch, dim, dstate).astype(np_dtype)
        
        # x: (batch, dim)
        x = np.random.randn(batch, dim).astype(np_dtype)
        
        # dt: (batch, dim)
        dt = np.random.uniform(0.1, 1.0, (batch, dim)).astype(np_dtype)
        
        # A: (dim, dstate)
        A = np.random.uniform(-2.0, -0.1, (dim, dstate)).astype(np_dtype)
        
        # B: (batch, n_groups, dstate)
        B = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # C: (batch, n_groups, dstate)
        C = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # D: (dim,)
        D = np.random.randn(dim).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, dim, dstate], device),  # state
            TensorType(dtype, [batch, dim], device),  # x
            TensorType(dtype, [batch, dim], device),  # dt
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate], device),  # B
            TensorType(dtype, [batch, n_groups, dstate], device),  # C
            TensorType(dtype, [dim], device),  # D
        ]
        
        graph = Graph(
            "test_selective_state_update_basic",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            state_val = graph.inputs[0].tensor
            x_val = graph.inputs[1].tensor
            dt_val = graph.inputs[2].tensor
            A_val = graph.inputs[3].tensor
            B_val = graph.inputs[4].tensor
            C_val = graph.inputs[5].tensor
            D_val = graph.inputs[6].tensor
            
            updated_state, output = selective_state_update_fn(
                state=state_val,
                x=x_val,
                dt=dt_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                dt_softplus=False,
            )
            graph.output(updated_state, output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(state).to(session.devices[0]),
            Tensor.from_numpy(x).to(session.devices[0]),
            Tensor.from_numpy(dt).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shapes
        assert len(results) == 2
        updated_state_tensor = results[0]
        output_tensor = results[1]
        assert updated_state_tensor.shape == (batch, dim, dstate)
        assert output_tensor.shape == (batch, dim)
    
    def test_selective_state_update_with_z(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective state update with gate tensor z."""
        batch, dim, dstate, n_groups = 2, 4, 2, 1
        dtype = DType.float32
        np_dtype = np.float32
        
        # state: (batch, dim, dstate)
        state = np.random.randn(batch, dim, dstate).astype(np_dtype)
        
        # x: (batch, dim)
        x = np.random.randn(batch, dim).astype(np_dtype)
        
        # dt: (batch, dim)
        dt = np.random.uniform(0.1, 1.0, (batch, dim)).astype(np_dtype)
        
        # A: (dim, dstate)
        A = np.random.uniform(-2.0, -0.1, (dim, dstate)).astype(np_dtype)
        
        # B: (batch, n_groups, dstate)
        B = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # C: (batch, n_groups, dstate)
        C = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # D: (dim,)
        D = np.random.randn(dim).astype(np_dtype)
        
        # z: (batch, dim)
        z = np.random.randn(batch, dim).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, dim, dstate], device),  # state
            TensorType(dtype, [batch, dim], device),  # x
            TensorType(dtype, [batch, dim], device),  # dt
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate], device),  # B
            TensorType(dtype, [batch, n_groups, dstate], device),  # C
            TensorType(dtype, [dim], device),  # D
            TensorType(dtype, [batch, dim], device),  # z
        ]
        
        graph = Graph(
            "test_selective_state_update_with_z",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            state_val = graph.inputs[0].tensor
            x_val = graph.inputs[1].tensor
            dt_val = graph.inputs[2].tensor
            A_val = graph.inputs[3].tensor
            B_val = graph.inputs[4].tensor
            C_val = graph.inputs[5].tensor
            D_val = graph.inputs[6].tensor
            z_val = graph.inputs[7].tensor
            
            updated_state, output = selective_state_update_fn(
                state=state_val,
                x=x_val,
                dt=dt_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                z=z_val,
                dt_softplus=False,
            )
            graph.output(updated_state, output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(state).to(session.devices[0]),
            Tensor.from_numpy(x).to(session.devices[0]),
            Tensor.from_numpy(dt).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
            Tensor.from_numpy(z).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shapes
        assert len(results) == 2
        updated_state_tensor = results[0]
        output_tensor = results[1]
        assert updated_state_tensor.shape == (batch, dim, dstate)
        assert output_tensor.shape == (batch, dim)
    
    def test_selective_state_update_with_dt_bias(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective state update with dt_bias."""
        batch, dim, dstate, n_groups = 2, 4, 2, 1
        dtype = DType.float32
        np_dtype = np.float32
        
        # state: (batch, dim, dstate)
        state = np.random.randn(batch, dim, dstate).astype(np_dtype)
        
        # x: (batch, dim)
        x = np.random.randn(batch, dim).astype(np_dtype)
        
        # dt: (batch, dim)
        dt = np.random.uniform(0.1, 1.0, (batch, dim)).astype(np_dtype)
        
        # A: (dim, dstate)
        A = np.random.uniform(-2.0, -0.1, (dim, dstate)).astype(np_dtype)
        
        # B: (batch, n_groups, dstate)
        B = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # C: (batch, n_groups, dstate)
        C = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # D: (dim,)
        D = np.random.randn(dim).astype(np_dtype)
        
        # dt_bias: (dim,)
        dt_bias = np.random.randn(dim).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, dim, dstate], device),  # state
            TensorType(dtype, [batch, dim], device),  # x
            TensorType(dtype, [batch, dim], device),  # dt
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate], device),  # B
            TensorType(dtype, [batch, n_groups, dstate], device),  # C
            TensorType(dtype, [dim], device),  # D
            TensorType(dtype, [dim], device),  # dt_bias
        ]
        
        graph = Graph(
            "test_selective_state_update_with_dt_bias",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            state_val = graph.inputs[0].tensor
            x_val = graph.inputs[1].tensor
            dt_val = graph.inputs[2].tensor
            A_val = graph.inputs[3].tensor
            B_val = graph.inputs[4].tensor
            C_val = graph.inputs[5].tensor
            D_val = graph.inputs[6].tensor
            dt_bias_val = graph.inputs[7].tensor
            
            updated_state, output = selective_state_update_fn(
                state=state_val,
                x=x_val,
                dt=dt_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                dt_bias=dt_bias_val,
                dt_softplus=False,
            )
            graph.output(updated_state, output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(state).to(session.devices[0]),
            Tensor.from_numpy(x).to(session.devices[0]),
            Tensor.from_numpy(dt).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
            Tensor.from_numpy(dt_bias).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shapes
        assert len(results) == 2
        updated_state_tensor = results[0]
        output_tensor = results[1]
        assert updated_state_tensor.shape == (batch, dim, dstate)
        assert output_tensor.shape == (batch, dim)
    
    def test_selective_state_update_dt_softplus(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective state update with dt_softplus=True."""
        batch, dim, dstate, n_groups = 2, 4, 2, 1
        dtype = DType.float32
        np_dtype = np.float32
        
        # state: (batch, dim, dstate)
        state = np.random.randn(batch, dim, dstate).astype(np_dtype)
        
        # x: (batch, dim)
        x = np.random.randn(batch, dim).astype(np_dtype)
        
        # dt: (batch, dim)
        dt = np.random.uniform(0.1, 1.0, (batch, dim)).astype(np_dtype)
        
        # A: (dim, dstate)
        A = np.random.uniform(-2.0, -0.1, (dim, dstate)).astype(np_dtype)
        
        # B: (batch, n_groups, dstate)
        B = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # C: (batch, n_groups, dstate)
        C = np.random.randn(batch, n_groups, dstate).astype(np_dtype)
        
        # D: (dim,)
        D = np.random.randn(dim).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, dim, dstate], device),  # state
            TensorType(dtype, [batch, dim], device),  # x
            TensorType(dtype, [batch, dim], device),  # dt
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate], device),  # B
            TensorType(dtype, [batch, n_groups, dstate], device),  # C
            TensorType(dtype, [dim], device),  # D
        ]
        
        graph = Graph(
            "test_selective_state_update_dt_softplus",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            state_val = graph.inputs[0].tensor
            x_val = graph.inputs[1].tensor
            dt_val = graph.inputs[2].tensor
            A_val = graph.inputs[3].tensor
            B_val = graph.inputs[4].tensor
            C_val = graph.inputs[5].tensor
            D_val = graph.inputs[6].tensor
            
            updated_state, output = selective_state_update_fn(
                state=state_val,
                x=x_val,
                dt=dt_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                dt_softplus=True,
            )
            graph.output(updated_state, output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(state).to(session.devices[0]),
            Tensor.from_numpy(x).to(session.devices[0]),
            Tensor.from_numpy(dt).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shapes
        assert len(results) == 2
        updated_state_tensor = results[0]
        output_tensor = results[1]
        assert updated_state_tensor.shape == (batch, dim, dstate)
        assert output_tensor.shape == (batch, dim)


class TestMambaInnerFn:
    """Tests for mamba_inner_fn."""
    
    def test_mamba_inner_basic(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test basic mamba_inner_fn forward pass."""
        batch, intermediate_size, seqlen, hidden_size = 2, 8, 16, 4
        d_state, delta_rank, conv_width = 2, 4, 4
        dtype = DType.float32
        np_dtype = np.float32
        
        # xz: (batch, 2 * intermediate_size, seqlen)
        xz = np.random.randn(batch, 2 * intermediate_size, seqlen).astype(np_dtype)
        
        # conv1d_weight: (intermediate_size, conv_width)
        conv1d_weight = np.random.randn(intermediate_size, conv_width).astype(np_dtype)
        
        # conv1d_bias: (intermediate_size,)
        conv1d_bias = np.random.randn(intermediate_size).astype(np_dtype)
        
        # x_proj_dim = delta_rank + 2 * d_state
        x_proj_dim = delta_rank + 2 * d_state
        # x_proj_weight: (x_proj_dim, intermediate_size)
        x_proj_weight = np.random.randn(x_proj_dim, intermediate_size).astype(np_dtype)
        
        # delta_proj_weight: (intermediate_size, delta_rank)
        delta_proj_weight = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)
        
        # out_proj_weight: (hidden_size, intermediate_size)
        out_proj_weight = np.random.randn(hidden_size, intermediate_size).astype(np_dtype)
        
        # out_proj_bias: (hidden_size,)
        out_proj_bias = np.random.randn(hidden_size).astype(np_dtype)
        
        # A: (intermediate_size, d_state)
        A = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
        
        # D: (intermediate_size,)
        D = np.random.randn(intermediate_size).astype(np_dtype)
        
        # delta_bias: (intermediate_size,)
        delta_bias = np.random.randn(intermediate_size).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, 2 * intermediate_size, seqlen], device),  # xz
            TensorType(dtype, [intermediate_size, conv_width], device),  # conv1d_weight
            TensorType(dtype, [intermediate_size], device),  # conv1d_bias
            TensorType(dtype, [x_proj_dim, intermediate_size], device),  # x_proj_weight
            TensorType(dtype, [intermediate_size, delta_rank], device),  # delta_proj_weight
            TensorType(dtype, [hidden_size, intermediate_size], device),  # out_proj_weight
            TensorType(dtype, [hidden_size], device),  # out_proj_bias
            TensorType(dtype, [intermediate_size, d_state], device),  # A
            TensorType(dtype, [intermediate_size], device),  # D
            TensorType(dtype, [intermediate_size], device),  # delta_bias
        ]
        
        graph = Graph(
            "test_mamba_inner_basic",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            xz_val = graph.inputs[0].tensor
            conv1d_weight_val = graph.inputs[1].tensor
            conv1d_bias_val = graph.inputs[2].tensor
            x_proj_weight_val = graph.inputs[3].tensor
            delta_proj_weight_val = graph.inputs[4].tensor
            out_proj_weight_val = graph.inputs[5].tensor
            out_proj_bias_val = graph.inputs[6].tensor
            A_val = graph.inputs[7].tensor
            D_val = graph.inputs[8].tensor
            delta_bias_val = graph.inputs[9].tensor
            
            output = mamba_inner_fn(
                xz=xz_val,
                conv1d_weight=conv1d_weight_val,
                conv1d_bias=conv1d_bias_val,
                x_proj_weight=x_proj_weight_val,
                delta_proj_weight=delta_proj_weight_val,
                out_proj_weight=out_proj_weight_val,
                out_proj_bias=out_proj_bias_val,
                A=A_val,
                D=D_val,
                delta_bias=delta_bias_val,
                delta_softplus=True,
            )
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(xz).to(session.devices[0]),
            Tensor.from_numpy(conv1d_weight).to(session.devices[0]),
            Tensor.from_numpy(conv1d_bias).to(session.devices[0]),
            Tensor.from_numpy(x_proj_weight).to(session.devices[0]),
            Tensor.from_numpy(delta_proj_weight).to(session.devices[0]),
            Tensor.from_numpy(out_proj_weight).to(session.devices[0]),
            Tensor.from_numpy(out_proj_bias).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
            Tensor.from_numpy(delta_bias).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, seqlen, hidden_size)
    
    def test_mamba_inner_basic_cpu(
        self, mamba_kernel_path: list[Path]
    ) -> None:
        """Test basic mamba_inner_fn forward pass on CPU."""
        # Force CPU device
        cpu_session = InferenceSession(devices=[CPU()])
        cpu_device = DeviceRef.CPU()
        
        batch, intermediate_size, seqlen, hidden_size = 2, 8, 16, 4
        d_state, delta_rank, conv_width = 2, 4, 4
        dtype = DType.float32
        np_dtype = np.float32
        
        # xz: (batch, 2 * intermediate_size, seqlen)
        xz = np.random.randn(batch, 2 * intermediate_size, seqlen).astype(np_dtype)
        
        # conv1d_weight: (intermediate_size, conv_width)
        conv1d_weight = np.random.randn(intermediate_size, conv_width).astype(np_dtype)
        
        # conv1d_bias: (intermediate_size,)
        conv1d_bias = np.random.randn(intermediate_size).astype(np_dtype)
        
        # x_proj_dim = delta_rank + 2 * d_state
        x_proj_dim = delta_rank + 2 * d_state
        # x_proj_weight: (x_proj_dim, intermediate_size)
        x_proj_weight = np.random.randn(x_proj_dim, intermediate_size).astype(np_dtype)
        
        # delta_proj_weight: (intermediate_size, delta_rank)
        delta_proj_weight = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)
        
        # out_proj_weight: (hidden_size, intermediate_size)
        out_proj_weight = np.random.randn(hidden_size, intermediate_size).astype(np_dtype)
        
        # out_proj_bias: (hidden_size,)
        out_proj_bias = np.random.randn(hidden_size).astype(np_dtype)
        
        # A: (intermediate_size, d_state)
        A = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
        
        # D: (intermediate_size,)
        D = np.random.randn(intermediate_size).astype(np_dtype)
        
        # delta_bias: (intermediate_size,)
        delta_bias = np.random.randn(intermediate_size).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, 2 * intermediate_size, seqlen], cpu_device),  # xz
            TensorType(dtype, [intermediate_size, conv_width], cpu_device),  # conv1d_weight
            TensorType(dtype, [intermediate_size], cpu_device),  # conv1d_bias
            TensorType(dtype, [x_proj_dim, intermediate_size], cpu_device),  # x_proj_weight
            TensorType(dtype, [intermediate_size, delta_rank], cpu_device),  # delta_proj_weight
            TensorType(dtype, [hidden_size, intermediate_size], cpu_device),  # out_proj_weight
            TensorType(dtype, [hidden_size], cpu_device),  # out_proj_bias
            TensorType(dtype, [intermediate_size, d_state], cpu_device),  # A
            TensorType(dtype, [intermediate_size], cpu_device),  # D
            TensorType(dtype, [intermediate_size], cpu_device),  # delta_bias
        ]
        
        graph = Graph(
            "test_mamba_inner_basic_cpu",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            xz_val = graph.inputs[0].tensor
            conv1d_weight_val = graph.inputs[1].tensor
            conv1d_bias_val = graph.inputs[2].tensor
            x_proj_weight_val = graph.inputs[3].tensor
            delta_proj_weight_val = graph.inputs[4].tensor
            out_proj_weight_val = graph.inputs[5].tensor
            out_proj_bias_val = graph.inputs[6].tensor
            A_val = graph.inputs[7].tensor
            D_val = graph.inputs[8].tensor
            delta_bias_val = graph.inputs[9].tensor
            
            output = mamba_inner_fn(
                xz=xz_val,
                conv1d_weight=conv1d_weight_val,
                conv1d_bias=conv1d_bias_val,
                x_proj_weight=x_proj_weight_val,
                delta_proj_weight=delta_proj_weight_val,
                out_proj_weight=out_proj_weight_val,
                out_proj_bias=out_proj_bias_val,
                A=A_val,
                D=D_val,
                delta_bias=delta_bias_val,
                delta_softplus=True,
            )
            graph.output(output)
        
        # Compile and execute on CPU
        compiled_model = cpu_session.load(graph)
        inputs = [
            Tensor.from_numpy(xz).to(cpu_session.devices[0]),
            Tensor.from_numpy(conv1d_weight).to(cpu_session.devices[0]),
            Tensor.from_numpy(conv1d_bias).to(cpu_session.devices[0]),
            Tensor.from_numpy(x_proj_weight).to(cpu_session.devices[0]),
            Tensor.from_numpy(delta_proj_weight).to(cpu_session.devices[0]),
            Tensor.from_numpy(out_proj_weight).to(cpu_session.devices[0]),
            Tensor.from_numpy(out_proj_bias).to(cpu_session.devices[0]),
            Tensor.from_numpy(A).to(cpu_session.devices[0]),
            Tensor.from_numpy(D).to(cpu_session.devices[0]),
            Tensor.from_numpy(delta_bias).to(cpu_session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, seqlen, hidden_size)
    
    def test_mamba_inner_simplified_gpu(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Simplified test to isolate GPU issue - test just selective_scan_fn with manually constructed inputs."""
        # Skip if not GPU
        if device.device_type != "gpu":
            pytest.skip("This test is for GPU only")
        
        batch, intermediate_size, seqlen = 2, 8, 16
        d_state, n_groups = 2, 1
        dtype = DType.float32
        np_dtype = np.float32
        
        # Create minimal inputs for selective_scan_fn
        # u: (batch, dim, seqlen)
        u = np.random.randn(batch, intermediate_size, seqlen).astype(np_dtype)
        # delta: (batch, dim, seqlen)
        delta = np.random.uniform(0.1, 1.0, (batch, intermediate_size, seqlen)).astype(np_dtype)
        # A: (dim, dstate)
        A = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
        # B: (batch, n_groups, dstate, seqlen)
        B = np.random.randn(batch, n_groups, d_state, seqlen).astype(np_dtype)
        # C: (batch, n_groups, dstate, seqlen)
        C = np.random.randn(batch, n_groups, d_state, seqlen).astype(np_dtype)
        # D: (dim,)
        D = np.random.randn(intermediate_size).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, intermediate_size, seqlen], device),  # u
            TensorType(dtype, [batch, intermediate_size, seqlen], device),  # delta
            TensorType(dtype, [intermediate_size, d_state], device),  # A
            TensorType(dtype, [batch, n_groups, d_state, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, d_state, seqlen], device),  # C
            TensorType(dtype, [intermediate_size], device),  # D
        ]
        
        graph = Graph(
            "test_mamba_inner_simplified_gpu",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            D_val = graph.inputs[5].tensor
            
            # Test just selective_scan_fn directly
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                delta_softplus=True,
            )
            # selective_scan_fn returns TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, intermediate_size, seqlen)

    def test_mamba_inner_tensor_construction_isolated(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective_scan_fn with tensors constructed the same way as mamba_inner_fn."""
        # Skip if not GPU
        if device.device_type != "gpu":
            pytest.skip("This test is for GPU only")

        batch, intermediate_size, seqlen = 2, 8, 16
        d_state, n_groups, delta_rank = 2, 1, 4
        dtype = DType.float32
        np_dtype = np.float32

        # Mimic mamba_inner_fn tensor construction
        # Create x_dbl: (batch * seqlen, x_proj_dim) where x_proj_dim = delta_rank + 2 * n_groups * d_state
        x_proj_dim = delta_rank + 2 * n_groups * d_state
        x_dbl_np = np.random.randn(batch * seqlen, x_proj_dim).astype(np_dtype)

        input_types = [
            TensorType(dtype, [batch * seqlen, x_proj_dim], device),  # x_dbl
        ]

        graph = Graph(
            "test_mamba_inner_tensor_construction_isolated",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            x_dbl_val = graph.inputs[0].tensor

            # Mimic the tensor construction from mamba_inner_fn
            # Split x_dbl like mamba_inner_fn does
            bc_dim_size = n_groups * d_state * 2
            dt, BC = ops.split(
                x_dbl_val,
                [delta_rank, bc_dim_size],
                axis=-1,
            )

            # Create delta_proj_weight for matrix multiplication
            delta_proj_weight_np = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)
            delta_proj_weight_val = ops.constant(delta_proj_weight_np, dtype=dtype, device=device)

            # Project delta: dt @ delta_proj_weight.T
            delta_flat = dt @ delta_proj_weight_val.T
            delta_val = ops.reshape(
                delta_flat,
                shape=[batch, intermediate_size, seqlen],
            )

            # Split BC into B and C
            B_flat, C_flat = ops.split(
                BC,
                [n_groups * d_state, n_groups * d_state],
                axis=-1,
            )

            # Reshape B and C like mamba_inner_fn
            B_val = ops.reshape(
                B_flat,
                shape=[batch, n_groups, d_state, seqlen],
            )
            C_val = ops.reshape(
                C_flat,
                shape=[batch, n_groups, d_state, seqlen],
            )

            # Create other tensors
            u_np = np.random.randn(batch, intermediate_size, seqlen).astype(np_dtype)
            u_val = ops.constant(u_np, dtype=dtype, device=device)

            A_np = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
            A_val = ops.constant(A_np, dtype=dtype, device=device)

            D_np = np.random.randn(intermediate_size).astype(np_dtype)
            D_val = ops.constant(D_np, dtype=dtype, device=device)

            # Test selective_scan_fn with these constructed tensors
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                z=None,
                delta_bias=None,
                delta_softplus=True,
            )
            # selective_scan_fn returns TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(x_dbl_np).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, intermediate_size, seqlen)

    def test_mamba_inner_causal_conv_isolated(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective_scan_fn with tensors created from causal_conv1d_fn output."""
        # Skip if not GPU
        if device.device_type != "gpu":
            pytest.skip("This test is for GPU only")

        batch, intermediate_size, seqlen = 2, 8, 16
        d_state, n_groups, delta_rank, conv_width = 2, 1, 4, 4
        dtype = DType.float32
        np_dtype = np.float32

        # Create inputs like mamba_inner_fn
        xz_np = np.random.randn(batch, 2 * intermediate_size, seqlen).astype(np_dtype)
        conv1d_weight_np = np.random.randn(intermediate_size, conv_width).astype(np_dtype)
        conv1d_bias_np = np.random.randn(intermediate_size).astype(np_dtype)
        x_proj_weight_np = np.random.randn(delta_rank + 2 * n_groups * d_state, intermediate_size).astype(np_dtype)
        delta_proj_weight_np = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)

        input_types = [
            TensorType(dtype, [batch, 2 * intermediate_size, seqlen], device),  # xz
            TensorType(dtype, [intermediate_size, conv_width], device),  # conv1d_weight
            TensorType(dtype, [intermediate_size], device),  # conv1d_bias
            TensorType(dtype, [delta_rank + 2 * n_groups * d_state, intermediate_size], device),  # x_proj_weight
            TensorType(dtype, [intermediate_size, delta_rank], device),  # delta_proj_weight
        ]

        graph = Graph(
            "test_mamba_inner_causal_conv_isolated",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            xz_val = graph.inputs[0].tensor
            conv1d_weight_val = graph.inputs[1].tensor
            conv1d_bias_val = graph.inputs[2].tensor
            x_proj_weight_val = graph.inputs[3].tensor
            delta_proj_weight_val = graph.inputs[4].tensor

            # Split xz like mamba_inner_fn does
            x, z = ops.split(xz_val, [intermediate_size, intermediate_size], axis=1)

            # Apply causal conv1d like mamba_inner_fn
            conv1d_out = causal_conv1d_fn(
                x,
                conv1d_weight_val,
                bias=conv1d_bias_val,
                algorithm="optimized",
                activation="silu",
            )

            # Reshape and project like mamba_inner_fn
            conv1d_out_flat = ops.reshape(
                conv1d_out,
                shape=[batch * seqlen, intermediate_size],
            )

            # Create x_dbl via matrix multiplication (this is the critical step)
            x_dbl = conv1d_out_flat @ x_proj_weight_val.T

            # Now do the same tensor construction as before
            bc_dim_size = n_groups * d_state * 2
            dt, BC = ops.split(
                x_dbl,
                [delta_rank, bc_dim_size],
                axis=-1,
            )

            # Project delta: dt @ delta_proj_weight.T
            delta_flat = dt @ delta_proj_weight_val.T
            delta_val = ops.reshape(
                delta_flat,
                shape=[batch, intermediate_size, seqlen],
            )

            # Split BC into B and C
            B_flat, C_flat = ops.split(
                BC,
                [n_groups * d_state, n_groups * d_state],
                axis=-1,
            )

            # Reshape B and C
            B_val = ops.reshape(
                B_flat,
                shape=[batch, n_groups, d_state, seqlen],
            )
            C_val = ops.reshape(
                C_flat,
                shape=[batch, n_groups, d_state, seqlen],
            )

            # Create other tensors
            A_np = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
            A_val = ops.constant(A_np, dtype=dtype, device=device)

            D_np = np.random.randn(intermediate_size).astype(np_dtype)
            D_val = ops.constant(D_np, dtype=dtype, device=device)

            # Force contiguous copy of conv1d_out by adding zero
            conv1d_out_contiguous = conv1d_out + ops.constant(0.0, dtype=conv1d_out.dtype, device=conv1d_out.device)

            # Test selective_scan_fn
            output = selective_scan_fn(
                u=conv1d_out_contiguous,  # Use contiguous version
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                z=None,
                delta_bias=None,
                delta_softplus=True,
            )
            assert not isinstance(output, tuple)
            graph.output(output)

        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(xz_np).to(session.devices[0]),
            Tensor.from_numpy(conv1d_weight_np).to(session.devices[0]),
            Tensor.from_numpy(conv1d_bias_np).to(session.devices[0]),
            Tensor.from_numpy(x_proj_weight_np).to(session.devices[0]),
            Tensor.from_numpy(delta_proj_weight_np).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, intermediate_size, seqlen)

    def test_mamba_inner_exact_construction(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test selective_scan_fn with exact same tensor construction as mamba_inner_fn."""
        # Skip if not GPU
        if device.device_type != "gpu":
            pytest.skip("This test is for GPU only")

        batch, intermediate_size, seqlen = 1, 4, 8  # Smaller dimensions to debug
        d_state, n_groups, delta_rank, conv_width = 2, 1, 4, 4
        dtype = DType.float32
        np_dtype = np.float32

        # Create inputs exactly like mamba_inner_fn
        xz_np = np.random.randn(batch, 2 * intermediate_size, seqlen).astype(np_dtype)
        conv1d_weight_np = np.random.randn(intermediate_size, conv_width).astype(np_dtype)
        conv1d_bias_np = np.random.randn(intermediate_size).astype(np_dtype)
        x_proj_weight_np = np.random.randn(delta_rank + 2 * n_groups * d_state, intermediate_size).astype(np_dtype)
        delta_proj_weight_np = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)

        input_types = [
            TensorType(dtype, [batch, 2 * intermediate_size, seqlen], device),  # xz
            TensorType(dtype, [intermediate_size, conv_width], device),  # conv1d_weight
            TensorType(dtype, [intermediate_size], device),  # conv1d_bias
            TensorType(dtype, [delta_rank + 2 * n_groups * d_state, intermediate_size], device),  # x_proj_weight
            TensorType(dtype, [intermediate_size, delta_rank], device),  # delta_proj_weight
        ]

        graph = Graph(
            "test_mamba_inner_exact_construction",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            xz_val = graph.inputs[0].tensor
            conv1d_weight_val = graph.inputs[1].tensor
            conv1d_bias_val = graph.inputs[2].tensor
            x_proj_weight_val = graph.inputs[3].tensor
            delta_proj_weight_val = graph.inputs[4].tensor

            # Exact same construction as mamba_inner_fn
            x, z = ops.split(xz_val, [intermediate_size, intermediate_size], axis=1)

            conv1d_out = causal_conv1d_fn(
                x,
                conv1d_weight_val,
                bias=conv1d_bias_val,
                algorithm="optimized",
                activation="silu",
            )

            conv1d_out_flat = ops.reshape(
                conv1d_out,
                shape=[batch * seqlen, intermediate_size],
            )

            x_dbl = conv1d_out_flat @ x_proj_weight_val.T

            # Compute n_groups, delta_rank, d_state like mamba_inner_fn
            x_proj_dim = x_proj_weight_val.shape[0]
            bc_dim = x_proj_dim - delta_rank
            n_groups_dim = bc_dim // (Dim(2) * Dim(d_state))
            n_groups = int(n_groups_dim) if n_groups_dim != 0 else 1

            dt, BC = ops.split(
                x_dbl,
                [delta_rank, n_groups * d_state * 2],
                axis=-1,
            )

            delta_flat = dt @ delta_proj_weight_val.T
            delta_val = ops.reshape(
                delta_flat,
                shape=[batch, intermediate_size, seqlen],
            )

            B_flat, C_flat = ops.split(
                BC,
                [n_groups * d_state, n_groups * d_state],
                axis=-1,
            )

            B_val = ops.reshape(
                B_flat,
                shape=[batch, n_groups, d_state, seqlen],
            )
            C_val = ops.reshape(
                C_flat,
                shape=[batch, n_groups, d_state, seqlen],
            )

            # Use conv1d_out as u (same as mamba_inner_fn)
            A_np = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
            A_val = ops.constant(A_np, dtype=dtype, device=device)

            D_np = np.random.randn(intermediate_size).astype(np_dtype)
            D_val = ops.constant(D_np, dtype=dtype, device=device)

            # Test selective_scan_fn with constructed tensors (should fail if kernel has issues with tensor layouts)
            output = selective_scan_fn(
                u=conv1d_out,  # Same as mamba_inner_fn
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                D=D_val,
                z=None,
                delta_bias=None,
                delta_softplus=True,
            )
            assert not isinstance(output, tuple)
            graph.output(output)

            # Alternative: Test with constant tensors to verify kernel works with correct shapes
            # u_const = ops.constant(np.random.randn(batch, intermediate_size, seqlen).astype(np_dtype), dtype=dtype, device=device)
            # delta_const = ops.constant(np.random.randn(batch, intermediate_size, seqlen).astype(np_dtype), dtype=dtype, device=device)
            # B_const = ops.constant(np.random.randn(batch, n_groups, d_state, seqlen).astype(np_dtype), dtype=dtype, device=device)
            # C_const = ops.constant(np.random.randn(batch, n_groups, d_state, seqlen).astype(np_dtype), dtype=dtype, device=device)
            #
            # output_const = selective_scan_fn(
            #     u=u_const,
            #     delta=delta_const,
            #     A=A_val,
            #     B=B_const,
            #     C=C_const,
            #     D=D_val,
            #     z=None,
            #     delta_bias=None,
            #     delta_softplus=True,
            # )
            # graph.output(output_const)

        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(xz_np).to(session.devices[0]),
            Tensor.from_numpy(conv1d_weight_np).to(session.devices[0]),
            Tensor.from_numpy(conv1d_bias_np).to(session.devices[0]),
            Tensor.from_numpy(x_proj_weight_np).to(session.devices[0]),
            Tensor.from_numpy(delta_proj_weight_np).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, intermediate_size, seqlen)

    def test_selective_scan_isolation(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test selective_scan_fn in complete isolation using the working pattern."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1

        # Create test data using the same create_test_data function as working tests
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
        ]

        # Use the same Graph construction pattern as the working test
        graph = Graph(
            "test_selective_scan_isolation",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor

            # Call selective_scan_fn the same way as the working test
            output = selective_scan_fn(
                u=u_val,
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        # Compile and execute the same way as the working test
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)

    def test_selective_scan_with_reshape(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test selective_scan_fn with ops.reshape() to isolate which operation breaks it."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1

        # Create test data
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
        ]

        graph = Graph(
            "test_selective_scan_with_reshape",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor

            # Add ops.reshape() - this might be the problematic operation
            u_reshaped = ops.reshape(u_val, u_val.shape)  # Reshape to same shape
            delta_reshaped = ops.reshape(delta_val, delta_val.shape)
            B_reshaped = ops.reshape(B_val, B_val.shape)
            C_reshaped = ops.reshape(C_val, C_val.shape)

            output = selective_scan_fn(
                u=u_reshaped,
                delta=delta_reshaped,
                A=A_val,
                B=B_reshaped,
                C=C_reshaped,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)

    def test_selective_scan_with_split(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test selective_scan_fn with ops.split() to isolate the problematic operation."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1

        # Create test data - create a tensor that can be split
        xz = np.random.randn(batch, dim * 2, seqlen).astype(np.float32)  # Shape for splitting
        delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )[1:]  # Skip u, we'll create it from split

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim * 2, seqlen], device),  # xz for splitting
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
        ]

        graph = Graph(
            "test_selective_scan_with_split",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            xz_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor

            # Add ops.split() - this might be the problematic operation
            x, y = ops.split(xz_val, [dim, dim], axis=1)

            output = selective_scan_fn(
                u=x,  # Use result of split
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(xz).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)

    def test_selective_scan_with_matmul(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test selective_scan_fn with matrix multiplication."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1

        # Create test data
        u, delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )

        # Create additional matrix for testing matmul
        weight = np.random.randn(dim, dim).astype(np.float32)

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim, seqlen], device),  # u
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
            TensorType(dtype, [dim, dim], device),  # weight for matmul
        ]

        graph = Graph(
            "test_selective_scan_with_matmul",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            u_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor
            weight_val = graph.inputs[5].tensor

            # Add matrix multiplication - this might be problematic
            # First reshape u for matmul: (batch, dim, seqlen) -> (batch * seqlen, dim)
            u_flat = ops.reshape(u_val, [batch * seqlen, dim])
            # Matrix multiply: (batch * seqlen, dim) @ (dim, dim) -> (batch * seqlen, dim)
            u_transformed = ops.matmul(u_flat, weight_val)
            # Reshape back: (batch * seqlen, dim) -> (batch, dim, seqlen)
            u_final = ops.reshape(u_transformed, [batch, dim, seqlen])

            output = selective_scan_fn(
                u=u_final,  # Use result of matmul operations
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(u).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
            Tensor.from_numpy(weight).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)

    def test_selective_scan_with_slice(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test selective_scan_fn with ops.slice_tensor()."""
        batch, dim, seqlen, dstate, n_groups = 2, 4, 8, 2, 1

        # Create test data - create a larger tensor for slicing
        large_tensor = np.random.randn(batch, dim * 2, seqlen).astype(np.float32)
        delta, A, B, C, D, z, delta_bias = create_test_data(
            batch, dim, seqlen, dstate, n_groups
        )[1:]  # Skip u

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, dim * 2, seqlen], device),  # large tensor for slicing
            TensorType(dtype, [batch, dim, seqlen], device),  # delta
            TensorType(dtype, [dim, dstate], device),  # A
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # B
            TensorType(dtype, [batch, n_groups, dstate, seqlen], device),  # C
        ]

        graph = Graph(
            "test_selective_scan_with_slice",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            large_val = graph.inputs[0].tensor
            delta_val = graph.inputs[1].tensor
            A_val = graph.inputs[2].tensor
            B_val = graph.inputs[3].tensor
            C_val = graph.inputs[4].tensor

            # Add ops.slice_tensor() - this might be problematic
            u_sliced = ops.slice_tensor(
                large_val,
                [slice(None), slice(0, dim), slice(None)],
            )

            output = selective_scan_fn(
                u=u_sliced,  # Use result of slice
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(large_tensor).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, dim, seqlen)

    def test_selective_scan_with_causal_conv1d(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test selective_scan_fn with causal_conv1d_fn output - this might be the problematic operation."""
        batch, intermediate_size, seqlen, d_state, conv_width = 2, 8, 16, 2, 4

        # Create test data
        x = np.random.randn(batch, intermediate_size, seqlen).astype(np.float32)
        conv_weight = np.random.randn(intermediate_size, conv_width).astype(np.float32)
        conv_bias = np.random.randn(intermediate_size).astype(np.float32)

        # Create other tensors
        delta = np.random.randn(batch, intermediate_size, seqlen).astype(np.float32)
        A = np.random.randn(intermediate_size, d_state).astype(np.float32)
        B = np.random.randn(batch, 1, d_state, seqlen).astype(np.float32)
        C = np.random.randn(batch, 1, d_state, seqlen).astype(np.float32)

        dtype = DType.float32
        input_types = [
            TensorType(dtype, [batch, intermediate_size, seqlen], device),  # x
            TensorType(dtype, [intermediate_size, conv_width], device),  # conv_weight
            TensorType(dtype, [intermediate_size], device),  # conv_bias
            TensorType(dtype, [batch, intermediate_size, seqlen], device),  # delta
            TensorType(dtype, [intermediate_size, d_state], device),  # A
            TensorType(dtype, [batch, 1, d_state, seqlen], device),  # B
            TensorType(dtype, [batch, 1, d_state, seqlen], device),  # C
        ]

        graph = Graph(
            "test_selective_scan_with_causal_conv1d",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )

        with graph:
            x_val = graph.inputs[0].tensor
            conv_weight_val = graph.inputs[1].tensor
            conv_bias_val = graph.inputs[2].tensor
            delta_val = graph.inputs[3].tensor
            A_val = graph.inputs[4].tensor
            B_val = graph.inputs[5].tensor
            C_val = graph.inputs[6].tensor

            # Add causal_conv1d_fn - this might be the problematic operation
            conv_out = causal_conv1d_fn(
                x_val,
                conv_weight_val,
                bias=conv_bias_val,
                # activation=None by default
            )

            output = selective_scan_fn(
                u=conv_out,  # Use result of causal_conv1d_fn
                delta=delta_val,
                A=A_val,
                B=B_val,
                C=C_val,
                delta_softplus=False,
            )
            # output is always TensorValue when return_last_state=False
            assert not isinstance(output, tuple)
            graph.output(output)

        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(x).to(session.devices[0]),
            Tensor.from_numpy(conv_weight).to(session.devices[0]),
            Tensor.from_numpy(conv_bias).to(session.devices[0]),
            Tensor.from_numpy(delta).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(B).to(session.devices[0]),
            Tensor.from_numpy(C).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, intermediate_size, seqlen)

    def test_mamba_inner_simplified(self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]) -> None:
        """Test simplified mamba_inner_fn that bypasses complex operations."""
        batch, intermediate_size, seqlen, hidden_size = 2, 8, 16, 4
        d_state, delta_rank, conv_width = 2, 4, 4
        dtype = DType.float32
        np_dtype = np.float32

        # Create test data - simplified version
        xz = np.random.randn(batch, 2 * intermediate_size, seqlen).astype(np_dtype)
        conv1d_weight = np.random.randn(intermediate_size, conv_width).astype(np_dtype)
        conv1d_bias = np.random.randn(intermediate_size).astype(np_dtype)
        x_proj_weight = np.random.randn(intermediate_size, intermediate_size).astype(np_dtype)
        delta_proj_weight = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)
        out_proj_weight = np.random.randn(hidden_size, intermediate_size).astype(np_dtype)
        out_proj_bias = np.random.randn(hidden_size).astype(np_dtype)
        A = np.random.randn(intermediate_size, d_state).astype(np_dtype)
        D = np.random.randn(intermediate_size).astype(np_dtype)
        delta_bias = np.random.randn(intermediate_size).astype(np_dtype)

        # Create graph
        with Graph("test_mamba_inner_simplified", custom_extensions=mamba_kernel_path) as graph:
            xz_tensor = ops.constant(xz, dtype=dtype, device=device)
            conv1d_weight_tensor = ops.constant(conv1d_weight, dtype=dtype, device=device)
            conv1d_bias_tensor = ops.constant(conv1d_bias, dtype=dtype, device=device)
            x_proj_weight_tensor = ops.constant(x_proj_weight, dtype=dtype, device=device)
            delta_proj_weight_tensor = ops.constant(delta_proj_weight, dtype=dtype, device=device)
            out_proj_weight_tensor = ops.constant(out_proj_weight, dtype=dtype, device=device)
            out_proj_bias_tensor = ops.constant(out_proj_bias, dtype=dtype, device=device)
            A_tensor = ops.constant(A, dtype=dtype, device=device)
            D_tensor = ops.constant(D, dtype=dtype, device=device)
            delta_bias_tensor = ops.constant(delta_bias, dtype=dtype, device=device)

            # Call simplified mamba_inner_fn
            result = mamba_inner_fn_simplified(
                xz_tensor,
                conv1d_weight_tensor,
                conv1d_bias_tensor,
                x_proj_weight_tensor,
                delta_proj_weight_tensor,
                out_proj_weight_tensor,
                out_proj_bias_tensor,
                A_tensor,
                D_tensor,
                delta_bias_tensor,
            )

        # Load and run
        compiled_model = session.load(graph)
        results = compiled_model.execute()

        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, seqlen, hidden_size)


class TestMambaInnerRef:
    """Tests for mamba_inner_ref."""
    
    def test_mamba_inner_ref_basic(
        self, session: InferenceSession, device: DeviceRef, mamba_kernel_path: list[Path]
    ) -> None:
        """Test basic mamba_inner_ref forward pass."""
        batch, intermediate_size, seqlen, hidden_size = 2, 8, 16, 4
        d_state, delta_rank, conv_width = 2, 4, 4
        dtype = DType.float32
        np_dtype = np.float32
        
        # Create test data
        xz = np.random.randn(batch, 2 * intermediate_size, seqlen).astype(np_dtype)
        conv1d_weight = np.random.randn(intermediate_size, conv_width).astype(np_dtype)
        conv1d_bias = np.random.randn(intermediate_size).astype(np_dtype)
        x_proj_dim = delta_rank + 2 * d_state
        x_proj_weight = np.random.randn(x_proj_dim, intermediate_size).astype(np_dtype)
        delta_proj_weight = np.random.randn(intermediate_size, delta_rank).astype(np_dtype)
        out_proj_weight = np.random.randn(hidden_size, intermediate_size).astype(np_dtype)
        out_proj_bias = np.random.randn(hidden_size).astype(np_dtype)
        A = np.random.uniform(-2.0, -0.1, (intermediate_size, d_state)).astype(np_dtype)
        D = np.random.randn(intermediate_size).astype(np_dtype)
        delta_bias = np.random.randn(intermediate_size).astype(np_dtype)
        
        input_types = [
            TensorType(dtype, [batch, 2 * intermediate_size, seqlen], device),  # xz
            TensorType(dtype, [intermediate_size, conv_width], device),  # conv1d_weight
            TensorType(dtype, [intermediate_size], device),  # conv1d_bias
            TensorType(dtype, [x_proj_dim, intermediate_size], device),  # x_proj_weight
            TensorType(dtype, [intermediate_size, delta_rank], device),  # delta_proj_weight
            TensorType(dtype, [hidden_size, intermediate_size], device),  # out_proj_weight
            TensorType(dtype, [hidden_size], device),  # out_proj_bias
            TensorType(dtype, [intermediate_size, d_state], device),  # A
            TensorType(dtype, [intermediate_size], device),  # D
            TensorType(dtype, [intermediate_size], device),  # delta_bias
        ]
        
        graph = Graph(
            "test_mamba_inner_ref_basic",
            input_types=input_types,
            custom_extensions=mamba_kernel_path,
        )
        
        with graph:
            xz_val = graph.inputs[0].tensor
            conv1d_weight_val = graph.inputs[1].tensor
            conv1d_bias_val = graph.inputs[2].tensor
            x_proj_weight_val = graph.inputs[3].tensor
            delta_proj_weight_val = graph.inputs[4].tensor
            out_proj_weight_val = graph.inputs[5].tensor
            out_proj_bias_val = graph.inputs[6].tensor
            A_val = graph.inputs[7].tensor
            D_val = graph.inputs[8].tensor
            delta_bias_val = graph.inputs[9].tensor
            
            output = mamba_inner_ref(
                xz=xz_val,
                conv1d_weight=conv1d_weight_val,
                conv1d_bias=conv1d_bias_val,
                x_proj_weight=x_proj_weight_val,
                delta_proj_weight=delta_proj_weight_val,
                out_proj_weight=out_proj_weight_val,
                out_proj_bias=out_proj_bias_val,
                A=A_val,
                D=D_val,
                delta_bias=delta_bias_val,
                delta_softplus=True,
            )
            graph.output(output)
        
        # Compile and execute
        compiled_model = session.load(graph)
        inputs = [
            Tensor.from_numpy(xz).to(session.devices[0]),
            Tensor.from_numpy(conv1d_weight).to(session.devices[0]),
            Tensor.from_numpy(conv1d_bias).to(session.devices[0]),
            Tensor.from_numpy(x_proj_weight).to(session.devices[0]),
            Tensor.from_numpy(delta_proj_weight).to(session.devices[0]),
            Tensor.from_numpy(out_proj_weight).to(session.devices[0]),
            Tensor.from_numpy(out_proj_bias).to(session.devices[0]),
            Tensor.from_numpy(A).to(session.devices[0]),
            Tensor.from_numpy(D).to(session.devices[0]),
            Tensor.from_numpy(delta_bias).to(session.devices[0]),
        ]
        results = compiled_model.execute(*inputs)
        
        # Verify output shape
        assert len(results) == 1
        output_tensor = results[0]
        assert output_tensor.shape == (batch, seqlen, hidden_size)
