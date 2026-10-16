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
"""Python wrapper for the fused gated group-RMSNorm kernel."""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops


def gated_group_rmsnorm(
    y: TensorValue,
    gate: TensorValue,
    norm_weight: TensorValue,
    eps: float,
    group_size: int,
) -> TensorValue:
    """Fuses the gated group-RMSNorm (HF ``Zamba2RMSNormGated`` with
    ``norm_before_gate=False``) into a single dispatch.

    Collapses ``cast(y->f32) -> silu(gate)*y -> group rms_norm ->
    *norm_weight -> cast`` into one op. ``y`` and ``gate`` are
    ``[N, intermediate]``; ``norm_weight`` is fp32 ``[intermediate]``.
    Each contiguous ``group_size`` slice of the intermediate axis is
    normalized independently. Returns the model dtype (``y.dtype``), so
    the downstream ``out_proj`` cast is a no-op.

    Args:
        y: The ``[N, intermediate]`` SSD scan output (model dtype).
        gate: The ``[N, intermediate]`` gate projection (any float dtype;
            may be a strided split view of the fused in-proj).
        norm_weight: The ``[intermediate]`` fp32 RMSNorm weight.
        eps: The epsilon inside ``rsqrt(mean_sq + eps)``.
        group_size: The width of each normalized group
            (``intermediate // n_groups``).

    Returns:
        The ``[N, intermediate]`` result in ``y.dtype``.

    The following example normalizes a 256-wide intermediate in groups of
    64:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU, Accelerator, Buffer, accelerator_count
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType
        from max.nn.state_space import gated_group_rmsnorm

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)
        num_tokens, intermediate, group_size = 4, 256, 64

        with Graph(
            "gated_group_rmsnorm_example",
            input_types=(
                TensorType(
                    DType.float32, [num_tokens, intermediate], device=device_ref
                ),
                TensorType(
                    DType.float32, [num_tokens, intermediate], device=device_ref
                ),
                TensorType(DType.float32, [intermediate], device=device_ref),
            ),
        ) as graph:
            y, gate, norm_weight = (value.tensor for value in graph.inputs)
            output = gated_group_rmsnorm(
                y, gate, norm_weight, eps=1e-5, group_size=group_size
            )
            graph.output(output)

        rng = np.random.default_rng(0)
        y_np = rng.standard_normal((num_tokens, intermediate), dtype=np.float32)
        gate_np = rng.standard_normal(
            (num_tokens, intermediate), dtype=np.float32
        )
        weight_np = rng.uniform(0.5, 1.5, intermediate).astype(np.float32)

        model = InferenceSession(devices=[device]).load(graph)
        inputs = [
            Buffer.from_numpy(x).to(device) for x in (y_np, gate_np, weight_np)
        ]
        result = model.execute(*inputs)[0].to_numpy()
        # result.shape is (4, 256), in y's dtype (float32).

    .. invisible-code-block: python

        assert result.shape == (num_tokens, intermediate)
        assert result.dtype == np.float32

        # gated = y * silu(gate), then RMS-normalize each group of 64 and
        # scale by norm_weight.
        y64, gate64 = y_np.astype(np.float64), gate_np.astype(np.float64)
        gated = y64 * gate64 / (1 + np.exp(-gate64))
        groups = gated.reshape(num_tokens, -1, group_size)
        rms = np.sqrt((groups**2).mean(axis=-1, keepdims=True) + 1e-5)
        expected = (groups / rms).reshape(num_tokens, intermediate) * weight_np
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    """
    device = y.device
    out_type = TensorType(y.dtype, [y.shape[0], y.shape[1]], device)
    # eps is a rank-0 fp32 CPU constant; the framework marshals it into the
    # kernel's ``Float32`` execute arg (the ``ops.rms_norm`` / fused-qk-rmsnorm
    # idiom).
    eps_c = ops.constant(eps, DType.float32, device=DeviceRef.CPU())
    results = ops.custom(
        "gated_group_rmsnorm",
        device,
        [y, gate, norm_weight, eps_c],
        [out_type],
        parameters={"group_size": group_size},
    )
    return cast(TensorValue, results[0])
