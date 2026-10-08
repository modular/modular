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
"""End-to-end check for the fused `mo.composite.mega_ffn_nvfp4` graph path.

Builds the two-leg NVFP4 MoE FFN chain -- `grouped_matmul_swiglu_nvfp4` (gate-up
"L1") followed by `grouped_matmul_block_scaled` (down "L2") -- as MAX graph ops
that share their EP routing tensors, with L1's two outputs feeding only L2. On
SM100 (`num_experts <= 64`, single-use intermediate, shared routing, bf16 out,
`mega-ffn-enable` default-true) the MegaFFN fusion rewrites the chain into a
single `mo.composite.mega_ffn_nvfp4`, mints the persistent `arrival_count`
scratch buffer itself (`mo.buffer.create` with a zero init value), and lowers
through the `builtin_kernels/mega_ffn.mojo` registration to
`mega_ffn_nvfp4_dispatch`.

This is a graph-level flow test: it asserts the graph compiles (the registration
type-checks against the composite op -- no "no kernel registered for
'mo.composite.mega_ffn_nvfp4'" / operand rank/dtype mismatch) and runs to a
correctly shaped/typed output, and that the swigluoai clamp selector and the
gate-up leg's per-row input scales reach the kernel through the fusion. That
the fusion fires at all is checked on the IR in
`test_mega_ffn_nvfp4_fusion_ir.py`. Deep numerical correctness (fused vs.
chained reference, byte-exact) is covered by the Mojo kernel test
`test_mega_ffn_nvfp4.mojo`.

DEPENDENCY: the composite op defs + the fusion pattern + the composite-emitting
`kernels.py` legs + the `mega_ffn.mojo` registration. Off SM100 the legs lower as
standalone leg kernels and the fusion never fires, so the target is gated to
B200.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _mega_ffn_graphs import build_graph, build_np_inputs
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from torch.utils.dlpack import from_dlpack


def _to_buffers(
    np_in: dict[str, np.ndarray], device: Accelerator
) -> list[Buffer]:
    """Copy inputs to device buffers in the order the graph expects.

    ``usage_stats`` stays on CPU per the graph signature.
    """

    def _gpu(arr: np.ndarray, dtype: DType) -> Buffer:
        buf = Buffer.from_dlpack(torch.from_numpy(arr.copy()))
        if dtype != DType.uint8 and arr.dtype == np.uint8:
            buf = buf.view(dtype)
        return buf.to(device)

    usage_stats_cpu = Buffer.from_dlpack(
        torch.from_numpy(np_in["usage_stats"].copy())
    )

    buffers = [
        _gpu(np_in["hidden"], DType.uint8),
        _gpu(np_in["w13"], DType.uint8),
        _gpu(np_in["a_scales"], DType.float8_e4m3fn),
        _gpu(np_in["b_scales13_pre"], DType.float8_e4m3fn),
        _gpu(np_in["down_w"], DType.uint8),
        _gpu(np_in["down_b_scales_pre"], DType.float8_e4m3fn),
        _gpu(np_in["expert_start"], DType.uint32),
        _gpu(np_in["a_scale_offsets"], DType.uint32),
        _gpu(np_in["expert_ids"], DType.int32),
        _gpu(np_in["es13"], DType.float32),
        _gpu(np_in["es_down"], DType.float32),
        usage_stats_cpu,
        _gpu(np_in["raw_input_scales"], DType.float32),
    ]
    if "row_scales" in np_in:
        # Stored as bf16 bit patterns; numpy has no bfloat16.
        bits = torch.from_numpy(np_in["row_scales"].copy())
        buffers.append(Buffer.from_dlpack(bits.view(torch.bfloat16)).to(device))
    return buffers


# E=16 = the EP-8 per-device shard of a 128-expert model; <= 64 (the fusion's
# hard scheduler limit). D = moe_dim, K1 = hidden_in, N2 = hidden_out.
#
# The kernel tiles both contraction dims by BK = 128 // size_of(uint8) = 128
# PACKED columns (= 256 unpacked NVFP4 elements) and the decode dispatch
# (M < 1024, mma_bn=8) uses k_group_size=2, so BOTH phases' k-iteration counts
# `ceildiv(K_unpacked/2, 128)` must be even (`mega_ffn_kernel.validate_config`).
# K1 = D = 512 (packed 256 -> 2 k-iters each) is the smallest shape that
# satisfies this for both legs.
@pytest.mark.parametrize(
    "label,E,M,D,K1,N2",
    [
        ("small", 16, 256, 512, 512, 256),
    ],
)
def test_mega_ffn_nvfp4_fusion_compiles_and_runs(
    label: str, E: int, M: int, D: int, K1: int, N2: int
) -> None:
    """Compile + run the chained MoE FFN; assert output shape/dtype.

    PASS == ``session.load`` succeeds (the registration type-checks against the
    composite op at instantiation: no "no kernel registered for
    'mo.composite.mega_ffn_nvfp4'", no operand rank/dtype/type mismatch, and the
    fusion-minted ``arrival_count`` buffer binds to the registration's
    ``MutableInputTensor`` arg) AND ``model.execute`` produces a ``(M, N2)`` bf16
    output. Numerics are validated in ``test_mega_ffn_nvfp4.mojo``.
    """
    rng = np.random.default_rng(1234)
    np_in, sf_dim_0 = build_np_inputs(E, M, D, K1, N2, rng)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    graph = build_graph(E, M, D, K1, N2, sf_dim_0, device_ref, cpu_ref)

    # THE validation: load runs MO -> MOGG and instantiates the registration.
    model = session.load(graph)

    inputs = _to_buffers(np_in, device)
    outputs = model.execute(*inputs)

    assert len(outputs) == 1, f"expected 1 output, got {len(outputs)}"
    out_np = from_dlpack(outputs[0]).cpu()
    assert tuple(out_np.shape) == (
        M,
        N2,
    ), f"output shape {tuple(out_np.shape)} != expected ({M}, {N2})"
    assert out_np.dtype == torch.bfloat16, (
        f"output dtype {out_np.dtype} != bfloat16"
    )
    print(
        (
            f"\n=== mega_ffn_nvfp4 fusion {label} "
            f"(E={E}, M={M}, D={D}, K1={K1}, N2={N2}) ===\n"
            f"  session.load + model.execute OK; output {tuple(out_np.shape)} "
            f"{out_np.dtype}"
        ),
        flush=True,
    )


def test_mega_ffn_nvfp4_fusion_clamp_reaches_kernel() -> None:
    """The swigluoai clamp selector reaches the fused kernel.

    Builds the SAME fused MoE FFN graph twice -- ``clamp_activation`` True vs
    False -- on identical inputs and asserts the outputs DIFFER. The selector
    rides as a Bool op attribute (emitter -> fusion -> registration comptime
    param); when set, the registration supplies swigluoai's canonical
    alpha/limit to the dispatch. Identical outputs would mean the selector never
    reached the kernel through the fusion.
    """
    E, M, D, K1, N2 = 16, 256, 512, 512, 256
    rng = np.random.default_rng(1234)
    np_in, sf_dim_0 = build_np_inputs(E, M, D, K1, N2, rng)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    def _load(clamp: bool) -> Model:
        return session.load(
            build_graph(
                E, M, D, K1, N2, sf_dim_0, device_ref, cpu_ref, clamp=clamp
            )
        )

    def _run(model: Model) -> torch.Tensor:
        outputs = model.execute(*_to_buffers(np_in, device))
        return from_dlpack(outputs[0]).cpu().to(torch.float32)

    # Load both before executing either: recording stops at the first execute.
    plain, clamped = _load(clamp=False), _load(clamp=True)
    out_plain = _run(plain)
    out_clamped = _run(clamped)

    assert tuple(out_plain.shape) == (M, N2)
    max_abs_diff = (out_plain - out_clamped).abs().max().item()
    assert max_abs_diff > 0.0, (
        "clamped and unclamped fused outputs are identical -- the swigluoai "
        "clamp selector did not reach the kernel through the fusion"
    )
    print(
        f"\n=== mega_ffn_nvfp4 clamp reaches kernel: "
        f"max|plain - clamped| = {max_abs_diff} (>0 => clamp applied) ===",
        flush=True,
    )


def test_mega_ffn_nvfp4_fusion_row_scales() -> None:
    """The gate-up leg's per-row input scales reach the fused kernel.

    Compiles the chain with ``a_row_scales`` on L1 twice: fused, and unfused
    (L1's packed output also returned, so the fusion bails). The fused output
    must match the unfused two-launch chain, and all-ones row scales must
    change the fused output.
    """
    E, M, D, K1, N2 = 16, 256, 512, 512, 256
    rng = np.random.default_rng(1234)
    np_in, sf_dim_0 = build_np_inputs(E, M, D, K1, N2, rng)

    def _bf16_bits(values: np.ndarray) -> np.ndarray:
        return (
            torch.from_numpy(values)
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
        )

    # The random-byte scale tiles saturate the NVFP4 intermediate, so a row
    # scale near 1 cannot move it. Log-uniform scales down to 2**-24 pull
    # rows back into range.
    np_rs = dict(np_in)
    np_rs["row_scales"] = _bf16_bits(np.exp2(rng.uniform(-24.0, 0.0, size=M)))
    np_ones = dict(np_in)
    np_ones["row_scales"] = _bf16_bits(np.ones(M))

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    def _load(keep_intermediate: bool) -> Model:
        return session.load(
            build_graph(
                E,
                M,
                D,
                K1,
                N2,
                sf_dim_0,
                device_ref,
                cpu_ref,
                row_scales=True,
                keep_intermediate=keep_intermediate,
            )
        )

    def _run(model: Model, inputs: dict[str, np.ndarray]) -> torch.Tensor:
        outputs = model.execute(*_to_buffers(inputs, device))
        return from_dlpack(outputs[0]).cpu()

    # Load both before executing either: recording stops at the first execute.
    fused_model = _load(keep_intermediate=False)
    unfused_model = _load(keep_intermediate=True)
    fused = _run(fused_model, np_rs)
    unfused = _run(unfused_model, np_rs)
    unit_row_scales = _run(fused_model, np_ones)

    assert torch.equal(fused, unfused), (
        "fused output with row scales differs from the unfused chain: max"
        f" |diff| = {(fused.float() - unfused.float()).abs().max().item()}"
    )
    assert not torch.equal(fused, unit_row_scales), (
        "row scales did not reach the fused kernel"
    )
