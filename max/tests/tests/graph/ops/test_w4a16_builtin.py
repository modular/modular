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

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, CPU, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.quantization import QuantizationConfig, QuantizationEncoding


GROUP_SIZE = 128


def _make_w4a16_qweight(k: int, n: int) -> np.ndarray:
    rows = np.arange(k, dtype=np.uint32)[:, None]
    packs = np.arange(n // 8, dtype=np.uint32)[None, :]
    qweight = np.zeros((k, n // 8), dtype=np.uint32)
    for lane in range(8):
        cols = packs * np.uint32(8) + np.uint32(lane)
        nibble = (rows * np.uint32(3) + cols * np.uint32(5)) & np.uint32(0xF)
        qweight |= nibble << np.uint32(lane * 4)
    return qweight.view(np.int32)


def _w4a16_to_gptq_bytes(qweight: np.ndarray, k: int, n: int) -> np.ndarray:
    raw = np.zeros((k // 8, n), dtype=np.uint32)
    qweight_u32 = qweight.view(np.uint32)
    cols = np.arange(n, dtype=np.uint32)
    for row in range(k):
        raw_row = row // 8
        k_shift = np.uint32((row & 7) * 4)
        packed = qweight_u32[row, cols // np.uint32(8)]
        nibble = (packed >> ((cols & np.uint32(7)) * np.uint32(4))) & np.uint32(
            0xF
        )
        raw[raw_row, :] |= nibble << k_shift
    return raw.view(np.uint8).reshape(k // 2, n)


def _dequant_w4a16(qweight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    k, n_packs = qweight.shape
    n = n_packs * 8
    group_size = k // scales.shape[0]
    rows = np.arange(k, dtype=np.uint32)[:, None]
    cols = np.arange(n, dtype=np.uint32)[None, :]
    packed = qweight.view(np.uint32)[rows, cols // np.uint32(8)]
    nibble = (packed >> ((cols & np.uint32(7)) * np.uint32(4))) & np.uint32(0xF)
    scale_rows = (np.arange(k) // group_size).astype(np.int64)
    return (nibble.astype(np.float32) - 8.0) * scales[scale_rows, :].astype(
        np.float32
    )


def _packed_qzeros(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, 0x88888888, dtype=np.uint32).view(np.int32)


def _build_gemm_model(
    name: str,
    device: Accelerator,
    m: int,
    k: int,
    n: int,
    sample_output: bool = False,
):
    device_ref = DeviceRef.from_device(device)
    groups = k // GROUP_SIZE
    with Graph(
        name,
        input_types=[
            TensorType(DType.float16, (m, k), device=device_ref),
            TensorType(DType.int32, (k, n // 8), device=device_ref),
            TensorType(DType.int32, (groups, n // 8), device=device_ref),
            TensorType(DType.float16, (groups, n), device=device_ref),
        ],
    ) as graph:
        a, qweight, qzeros, scales = graph.inputs
        out = ops.custom(
            name="gemm_w4a16_fp16",
            device=device_ref,
            values=[a.tensor, qweight.tensor, qzeros.tensor, scales.tensor],
            out_types=[TensorType(DType.float16, (m, n), device=device_ref)],
        )[0].tensor
        if sample_output:
            row_indices = ops.constant(
                sorted({0, m - 1}), DType.int64, device=device_ref
            )
            col_indices = ops.constant(
                sorted({0, n // 2, n - 1}), DType.int64, device=device_ref
            )
            out = ops.gather(
                ops.gather(out, row_indices, axis=0), col_indices, axis=1
            )
        graph.output(out)

    return InferenceSession(devices=[device]).load(graph)


def _build_composed_model(
    name: str,
    device: Accelerator,
    m: int,
    k: int,
    n: int,
):
    device_ref = DeviceRef.from_device(device)
    groups = k // GROUP_SIZE
    with Graph(
        name,
        input_types=[
            TensorType(DType.float16, (m, k), device=device_ref),
            TensorType(DType.uint8, (k // 2, n), device=device_ref),
            TensorType(DType.uint8, (groups * 2, n), device=device_ref),
        ],
    ) as graph:
        a, qweight_bytes, scales_bytes = graph.inputs
        qweight, qzeros, scales = (
            value.tensor
            for value in ops.custom(
                name="gptq_to_w4a16",
                device=device_ref,
                values=[qweight_bytes.tensor, scales_bytes.tensor],
                out_types=[
                    TensorType(DType.int32, (k, n // 8), device=device_ref),
                    TensorType(DType.int32, (groups, n // 8), device=device_ref),
                    TensorType(DType.float16, (groups, n), device=device_ref),
                ],
            )
        )
        out = ops.custom(
            name="gemm_w4a16_fp16",
            device=device_ref,
            values=[a.tensor, qweight, qzeros, scales],
            out_types=[TensorType(DType.float16, (m, n), device=device_ref)],
        )[0].tensor
        graph.output(out)

    return InferenceSession(devices=[device]).load(graph)


def _build_qmatmul_model(
    name: str,
    device: Accelerator,
    m: int,
    k: int,
    n: int,
):
    device_ref = DeviceRef.from_device(device)
    groups = k // GROUP_SIZE
    with Graph(
        name,
        input_types=[
            TensorType(DType.float16, (m, k), device=device_ref),
            TensorType(DType.uint8, (k // 2 + groups * 2, n), device=device_ref),
        ],
    ) as graph:
        a, weight = graph.inputs
        out = ops.qmatmul(
            QuantizationEncoding.GPTQ,
            QuantizationConfig(
                quant_method="gptq",
                bits=4,
                group_size=GROUP_SIZE,
                desc_act=False,
                sym=True,
            ),
            a.tensor,
            weight.tensor,
        )
        graph.output(out)

    return InferenceSession(devices=[device]).load(graph)


def test_w4a16_builtin_ops_do_not_need_custom_extensions() -> None:
    if accelerator_count() == 0:
        pytest.skip("W4A16 builtin op test requires a GPU")
    _run_builtin_accuracy()


def _run_builtin_accuracy() -> None:
    print("w4a16 builtin accuracy: m=8 k=128 n=64", flush=True)
    device = Accelerator()
    m, k, n = 8, 128, 64
    groups = k // GROUP_SIZE
    rng = np.random.default_rng(0)
    a_np = rng.normal(0, 0.25, size=(m, k)).astype(np.float16)
    qweight_np = _make_w4a16_qweight(k, n)
    qweight_bytes_np = _w4a16_to_gptq_bytes(qweight_np, k, n)
    scales_np = rng.normal(0, 0.1, size=(groups, n)).astype(np.float16)
    scales_bytes_np = scales_np.view(np.uint8).reshape(groups * 2, n)
    ref_np = (a_np.astype(np.float32) @ _dequant_w4a16(qweight_np, scales_np)).astype(
        np.float16
    )

    model = _build_composed_model("w4a16_builtin_composed_accuracy", device, m, k, n)
    result = model.execute(
        Buffer.from_numpy(a_np).to(device),
        Buffer.from_numpy(qweight_bytes_np).to(device),
        Buffer.from_numpy(scales_bytes_np).to(device),
    )[0]

    np.testing.assert_allclose(
        result.to(CPU()).to_numpy(), ref_np, rtol=5e-2, atol=5e-2
    )

    print("w4a16 qmatmul accuracy: m=8 k=128 n=64", flush=True)
    qmatmul_model = _build_qmatmul_model(
        "w4a16_builtin_qmatmul_accuracy", device, m, k, n
    )
    packed_weight_np = np.concatenate((qweight_bytes_np, scales_bytes_np), axis=0)
    qmatmul_result = qmatmul_model.execute(
        Buffer.from_numpy(a_np).to(device),
        Buffer.from_numpy(packed_weight_np).to(device),
    )[0]

    np.testing.assert_allclose(
        qmatmul_result.to(CPU()).to_numpy(), ref_np, rtol=5e-2, atol=5e-2
    )


def test_w4a16_builtin_llama_shape_sweep() -> None:
    if os.environ.get("MAX_W4A16_LLAMA_SHAPE_SWEEP") != "1":
        pytest.skip("set MAX_W4A16_LLAMA_SHAPE_SWEEP=1 to run the full shape sweep")
    if accelerator_count() == 0:
        pytest.skip("W4A16 Llama shape sweep requires a GPU")
    _run_llama_shape_sweep()


def _run_llama_shape_sweep() -> None:
    device = Accelerator()
    max_m = int(os.environ.get("MAX_W4A16_SWEEP_MAX_M", "65536"))
    m_values = [1 << i for i in range(17) if (1 << i) <= max_m]
    all_shapes = [
        ("kv_proj", 4096, 1024),
        ("q_o_proj", 4096, 4096),
        ("gate_up_proj", 4096, 14336),
        ("down_proj", 14336, 4096),
    ]
    shape_filter = os.environ.get("MAX_W4A16_SWEEP_SHAPES")
    if shape_filter:
        wanted = set(shape_filter.split(","))
        shapes = [shape for shape in all_shapes if shape[0] in wanted]
    else:
        shapes = all_shapes

    for shape_name, k, n in shapes:
        print(f"w4a16 sweep shape={shape_name} k={k} n={n}", flush=True)
        groups = k // GROUP_SIZE
        qweight_np = _make_w4a16_qweight(k, n)
        qzeros_np = _packed_qzeros((groups, n // 8))
        scales_np = np.full((groups, n), 0.03125, dtype=np.float16)
        qweight_buf = Buffer.from_numpy(qweight_np).to(device)
        qzeros_buf = Buffer.from_numpy(qzeros_np).to(device)
        scales_buf = Buffer.from_numpy(scales_np).to(device)

        for m in m_values:
            print(f"w4a16 sweep shape={shape_name} m={m}", flush=True)
            model = _build_gemm_model(
                f"w4a16_llama_shape_sweep_{shape_name}_{m}_{k}_{n}",
                device,
                m,
                k,
                n,
                sample_output=True,
            )
            a = torch.zeros((m, k), dtype=torch.float16)
            a[:, 0] = 1.0
            result = model.execute(
                Buffer.from_dlpack(a).to(device),
                qweight_buf,
                qzeros_buf,
                scales_buf,
            )[0]
            result_np = result.to(CPU()).to_numpy()
            assert result_np.shape == (1 if m == 1 else 2, 3)
            assert np.isfinite(result_np).all()


def main() -> None:
    if accelerator_count() == 0:
        raise RuntimeError("W4A16 Llama shape sweep requires a GPU")
    _run_builtin_accuracy()
    _run_llama_shape_sweep()


if __name__ == "__main__":
    main()
