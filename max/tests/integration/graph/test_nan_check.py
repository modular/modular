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
"""Integration tests for NaN/Inf detection via max-debug.nan-check.

These tests verify the end-to-end behavior of the NaN checking feature:
- NaN detection triggers a fatal error with a diagnostic message
- Inf detection triggers a fatal error with a diagnostic message
- Clean tensors pass through without error
- No overhead when the feature is disabled

Tests are parametrized over device (CPU and GPU) since the NaN check kernel
has meaningfully different code paths: CPU uses a sequential scan, GPU uses
a parallel reduction with shared memory and atomics.
"""

import os
import re
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_SCRIPTS_DIR = Path(__file__).parent / "nan_check_scripts"

_has_gpu = accelerator_count() > 0


@pytest.fixture(autouse=True)
def _reset_debug_config() -> Generator[None, None, None]:
    """Reset the DebugConfig singleton after each in-process test."""
    yield
    InferenceSession.debug.reset()


class DeviceConfig(NamedTuple):
    name: str
    device: Device
    device_ref: DeviceRef


@pytest.fixture(params=["cpu", "gpu"])
def device_config(request: pytest.FixtureRequest) -> DeviceConfig:
    if request.param == "gpu":
        if not _has_gpu:
            pytest.skip("No GPU available")
        return DeviceConfig("gpu", Accelerator(), DeviceRef.GPU())
    return DeviceConfig("cpu", CPU(), DeviceRef.CPU())


def _build_div_graph(device_ref: DeviceRef) -> Graph:
    """Build a simple graph that divides two tensors element-wise.

    Used to produce NaN (0/0) or Inf (1/0) depending on inputs.
    """
    input_type = TensorType(dtype=DType.float32, shape=[4], device=device_ref)
    with Graph("div_graph", input_types=[input_type, input_type]) as graph:
        result = ops.div(graph.inputs[0], graph.inputs[1])  # type: ignore[arg-type]
        graph.output(result)
    return graph


def _build_identity_graph(device_ref: DeviceRef) -> Graph:
    """Build a simple identity graph (pass-through) for clean tensor tests."""
    input_type = TensorType(dtype=DType.float32, shape=[4], device=device_ref)
    with Graph("identity_graph", input_types=[input_type]) as graph:
        graph.output(graph.inputs[0])
    return graph


def _build_relu_graph(device_ref: DeviceRef) -> Graph:
    """Build a simple relu graph for testing with valid outputs."""
    input_type = TensorType(
        dtype=DType.float32, shape=[4, 5], device=device_ref
    )
    with Graph("relu_graph", input_types=[input_type]) as graph:
        result = ops.relu(graph.inputs[0])
        graph.output(result)
    return graph


def _run_script_in_subprocess(
    script_path: Path,
    env_vars: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python script file in a subprocess with optional env vars.

    NaN/Inf detection causes a fatal abort, which cannot be caught in-process.
    We run detection tests in a subprocess and inspect exit code + stderr.
    """
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


class TestNanCheckDetectsNan:
    """Verify that NaN values in graph outputs trigger a fatal error."""

    def test_nan_from_zero_div_zero(self, device_config: DeviceConfig) -> None:
        """0.0 / 0.0 produces NaN; nan_check should abort with diagnostic."""
        result = _run_script_in_subprocess(
            _SCRIPTS_DIR / "produce_nan.py",
            env_vars={
                "MODULAR_DEBUG": "nan-check",
                "NAN_CHECK_TEST_DEVICE": device_config.name,
            },
        )
        # The process should exit with non-zero (fatal abort)
        assert result.returncode != 0, (
            f"Expected non-zero exit code when NaN detected, "
            f"got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # The error message should contain NaN/Inf diagnostic info
        combined_output = result.stdout + result.stderr
        assert "NaN" in combined_output or "nan" in combined_output.lower(), (
            f"Expected 'NaN' in error output.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_nan_error_message_format(
        self, device_config: DeviceConfig
    ) -> None:
        """Verify the error message contains shape, dtype, and NaN count."""
        result = _run_script_in_subprocess(
            _SCRIPTS_DIR / "produce_nan.py",
            env_vars={
                "MODULAR_DEBUG": "nan-check",
                "NAN_CHECK_TEST_DEVICE": device_config.name,
            },
        )
        assert result.returncode != 0
        combined_output = result.stdout + result.stderr

        # Validate error message format:
        # "NaN/Inf detected in '<label>' (<type_str>): <N> NaN, <M> Inf"
        assert re.search(r"NaN/Inf detected in '.*'", combined_output), (
            f"Missing 'NaN/Inf detected in' label.\nOutput: {combined_output}"
        )

        assert re.search(r"\d+ NaN", combined_output), (
            f"Missing NaN count in error message.\nOutput: {combined_output}"
        )

        assert re.search(r"\d+ Inf", combined_output), (
            f"Missing Inf count in error message.\nOutput: {combined_output}"
        )

        assert re.search(r"f32", combined_output), (
            f"Missing dtype in error message.\nOutput: {combined_output}"
        )


class TestNanCheckPassesCleanTensor:
    """Verify that valid tensors pass through nan_check without error."""

    def test_clean_tensor_identity(
        self,
        device_config: DeviceConfig,
    ) -> None:
        """An identity graph with valid values should complete normally."""
        InferenceSession.debug.nan_check = True

        session = InferenceSession(devices=[device_config.device])
        graph = _build_identity_graph(device_config.device_ref)
        model = session.load(graph)

        clean_input = Buffer.from_numpy(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ).to(model.input_devices[0])

        output = model.execute(clean_input)
        assert len(output) == 1
        assert isinstance(output[0], Buffer)
        assert np.allclose(
            output[0].to_numpy(),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

    def test_clean_tensor_relu(
        self,
        device_config: DeviceConfig,
    ) -> None:
        """A relu graph with valid values should complete normally."""
        InferenceSession.debug.nan_check = True

        session = InferenceSession(devices=[device_config.device])
        graph = _build_relu_graph(device_config.device_ref)
        model = session.load(graph)

        clean_input = Buffer.from_numpy(np.ones((4, 5), dtype=np.float32)).to(
            model.input_devices[0]
        )

        output = model.execute(clean_input)
        assert len(output) == 1
        assert isinstance(output[0], Buffer)
        # relu(1.0) = 1.0
        assert np.allclose(output[0].to_numpy(), np.ones((4, 5)))

    def test_clean_division(
        self,
        device_config: DeviceConfig,
    ) -> None:
        """A division that does NOT produce NaN/Inf should pass cleanly."""
        InferenceSession.debug.nan_check = True

        session = InferenceSession(devices=[device_config.device])
        graph = _build_div_graph(device_config.device_ref)
        model = session.load(graph)

        numerator = Buffer.from_numpy(
            np.array([4.0, 6.0, 8.0, 10.0], dtype=np.float32)
        ).to(model.input_devices[0])
        denominator = Buffer.from_numpy(
            np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ).to(model.input_devices[0])

        output = model.execute(numerator, denominator)
        assert len(output) == 1
        assert isinstance(output[0], Buffer)
        assert np.allclose(
            output[0].to_numpy(),
            np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        )


class TestNanCheckDetectsInf:
    """Verify that Inf values in graph outputs trigger a fatal error."""

    def test_inf_from_div_by_zero(self, device_config: DeviceConfig) -> None:
        """1.0 / 0.0 produces Inf; nan_check should abort with diagnostic."""
        result = _run_script_in_subprocess(
            _SCRIPTS_DIR / "produce_inf.py",
            env_vars={
                "MODULAR_DEBUG": "nan-check",
                "NAN_CHECK_TEST_DEVICE": device_config.name,
            },
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit code when Inf detected, "
            f"got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined_output = result.stdout + result.stderr
        assert "Inf" in combined_output or "inf" in combined_output.lower(), (
            f"Expected 'Inf' in error output.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_inf_error_message_has_count(
        self, device_config: DeviceConfig
    ) -> None:
        """Verify the Inf diagnostic includes the correct Inf count."""
        result = _run_script_in_subprocess(
            _SCRIPTS_DIR / "produce_inf.py",
            env_vars={
                "MODULAR_DEBUG": "nan-check",
                "NAN_CHECK_TEST_DEVICE": device_config.name,
            },
        )
        assert result.returncode != 0
        combined_output = result.stdout + result.stderr
        # Exactly 2 Inf values (indices 0 and 1)
        assert re.search(r"2 Inf", combined_output), (
            f"Expected '2 Inf' in error message.\nOutput: {combined_output}"
        )
        # 0 NaN values in this case
        assert re.search(r"0 NaN", combined_output), (
            f"Expected '0 NaN' in error message.\nOutput: {combined_output}"
        )

    def test_mixed_nan_and_inf(self, device_config: DeviceConfig) -> None:
        """Verify diagnostic reports both NaN and Inf counts together."""
        result = _run_script_in_subprocess(
            _SCRIPTS_DIR / "produce_mixed.py",
            env_vars={
                "MODULAR_DEBUG": "nan-check",
                "NAN_CHECK_TEST_DEVICE": device_config.name,
            },
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit code with mixed NaN+Inf, "
            f"got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined_output = result.stdout + result.stderr
        # Should report both NaN and Inf counts
        assert re.search(r"1 NaN", combined_output), (
            f"Expected '1 NaN' in error message.\nOutput: {combined_output}"
        )
        assert re.search(r"1 Inf", combined_output), (
            f"Expected '1 Inf' in error message.\nOutput: {combined_output}"
        )
        assert re.search(r"f32", combined_output), (
            f"Expected 'f32' in error message.\nOutput: {combined_output}"
        )


class TestNanCheckNoOverheadWhenDisabled:
    """Verify no nan_check ops are inserted when the flag is not set."""

    def test_nan_produces_nan_without_abort(
        self, device_config: DeviceConfig
    ) -> None:
        """Without max-debug.nan-check enabled, 0/0 should produce NaN silently."""
        session = InferenceSession(devices=[device_config.device])
        graph = _build_div_graph(device_config.device_ref)
        model = session.load(graph)

        # 0/0 = NaN, but no abort expected since nan check is disabled
        numerator = Buffer.from_numpy(
            np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        ).to(model.input_devices[0])
        denominator = Buffer.from_numpy(
            np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        ).to(model.input_devices[0])

        output = model.execute(numerator, denominator)
        assert len(output) == 1
        assert isinstance(output[0], Buffer)
        result = output[0].to_numpy()
        # Should have NaN values but no abort
        assert np.isnan(result[0])
        assert np.isnan(result[2])
        assert result[1] == 1.0
        assert result[3] == 2.0

    def test_inf_produces_inf_without_abort(
        self, device_config: DeviceConfig
    ) -> None:
        """Without max-debug.nan-check enabled, 1/0 should produce Inf silently."""
        session = InferenceSession(devices=[device_config.device])
        graph = _build_div_graph(device_config.device_ref)
        model = session.load(graph)

        numerator = Buffer.from_numpy(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ).to(model.input_devices[0])
        denominator = Buffer.from_numpy(
            np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        ).to(model.input_devices[0])

        output = model.execute(numerator, denominator)
        assert len(output) == 1
        result = output[0].to_numpy()
        assert np.isinf(result[0])
        assert np.isinf(result[1])
        assert result[2] == 3.0
        assert result[3] == 4.0

    def test_disabled_when_flag_off(
        self,
        device_config: DeviceConfig,
    ) -> None:
        """Ensure nan checking is off when `InferenceSession.debug.nan_check` is False."""
        InferenceSession.debug.nan_check = False

        session = InferenceSession(devices=[device_config.device])
        graph = _build_div_graph(device_config.device_ref)
        model = session.load(graph)

        # 0/0 = NaN, but should NOT abort since env var is unset
        numerator = Buffer.from_numpy(
            np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        ).to(model.input_devices[0])
        denominator = Buffer.from_numpy(
            np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        ).to(model.input_devices[0])

        output = model.execute(numerator, denominator)
        assert len(output) == 1
        result = output[0].to_numpy()
        assert np.isnan(result[0])
