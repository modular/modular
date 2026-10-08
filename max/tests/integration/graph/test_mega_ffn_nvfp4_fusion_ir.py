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
"""The MegaFFN fusion fires: the end-of-MO IR carries the fused op.

Dumps the graph-compiler MO IR (``max-debug.ir-output-dir``) while compiling
the chained two-leg graphs from ``_mega_ffn_graphs``, then asserts the raw
``.mo.mlir`` still shows the two leg composites while the post-MO
``.mo-pre-mogg.mlir`` (after the MO ``PatternFusion`` pass) has collapsed them
into a single ``mo.composite.mega_ffn_nvfp4``. The execution checks in
``test_mega_ffn_nvfp4_fusion_gpu.py`` would also pass on the unfused two-kernel
fallback; this proves the fusion pattern fired on the graphs the real
``kernels.py`` emitters build.

Compiling is all these need, so they run on a virtual SM100 device. Virtual-
device mode latches process-wide at the first device creation, so the knobs are
set at import and this needs a target of its own.
"""

from __future__ import annotations

from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(1)

from pathlib import Path

from _mega_ffn_graphs import SF_MN_GROUP_SIZE, build_graph, build_graph_mxfp8
from max.driver import Accelerator
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph

E, M, D, K1, N2 = 16, 256, 512, 512, 256
SF_DIM_0 = M // SF_MN_GROUP_SIZE + E


def _compile_ir(graph: Graph, ir_dir: Path) -> tuple[str, str]:
    """Compiles ``graph`` and returns its raw and post-MO IR dumps."""
    session = InferenceSession(devices=[Accelerator()])
    session.debug.ir_output_dir = str(ir_dir)
    session.load(graph)

    def _read(pattern: str) -> str:
        paths = sorted(ir_dir.glob(pattern))
        assert paths, f"no IR dump matched {pattern} in {ir_dir}"
        return "\n".join(p.read_text() for p in paths)

    return _read("*.mo.mlir"), _read("*.mo-pre-mogg.mlir")


def _op_defs(text: str, name: str) -> int:
    # Count OP DEFINITIONS (``... = mo.composite.<name>(``), not bare substrings:
    # the op name also appears inside `loc(".../grouped_matmul_swiglu_nvfp4.mojo")`
    # kernel-source references, which survive even after the op itself fuses.
    return text.count(f"= mo.composite.{name}(")


def _assert_fused(raw_mo: str, post_mo: str) -> None:
    # The graph the emitters built presents both leg composites to the fusion.
    assert _op_defs(raw_mo, "grouped_matmul_swiglu_nvfp4") > 0
    assert _op_defs(raw_mo, "grouped_matmul_block_scaled") > 0

    # After the MO PatternFusion pass the legs are gone, replaced by the fused
    # op minting its arrival_count buffer -- i.e. MegaFFNNvfp4Pattern fired.
    assert _op_defs(post_mo, "mega_ffn_nvfp4") > 0, (
        "MegaFFN fusion did NOT fire (no mega_ffn_nvfp4 in MO IR)"
    )
    assert _op_defs(post_mo, "grouped_matmul_swiglu_nvfp4") == 0, (
        "L1 leg op survived -- fusion did not consume it"
    )
    assert _op_defs(post_mo, "grouped_matmul_block_scaled") == 0, (
        "L2 leg op survived -- fusion did not consume it"
    )
    assert "mo.buffer.create" in post_mo, (
        "fused op did not mint its persistent arrival_count buffer"
    )


def test_mega_ffn_nvfp4_fusion_fires_in_mo_ir(tmp_path: Path) -> None:
    graph = build_graph(
        E, M, D, K1, N2, SF_DIM_0, DeviceRef.GPU(), DeviceRef.CPU()
    )
    _assert_fused(*_compile_ir(graph, tmp_path))


def test_mega_ffn_nvfp4_fusion_keeps_row_scales(tmp_path: Path) -> None:
    """The gate-up leg's per-row input scales survive onto the fused op."""
    graph = build_graph(
        E,
        M,
        D,
        K1,
        N2,
        SF_DIM_0,
        DeviceRef.GPU(),
        DeviceRef.CPU(),
        row_scales=True,
    )
    raw_mo, post_mo = _compile_ir(graph, tmp_path)
    _assert_fused(raw_mo, post_mo)
    assert "has_gate_up_a_row_scales = true" in post_mo, (
        "fused op lost the gate-up row scales"
    )


def test_mega_ffn_mxfp8_clamped_swiglu_fusion_fires_in_mo_ir(
    tmp_path: Path,
) -> None:
    """MXFP8 + clamped SwiGLU (swigluoai) also fuses -- the MiniMax-M3 shape.

    The two leg composites are dtype-agnostic (``MO_Tensor`` operands), so the
    SAME ``MegaFFNNvfp4Pattern`` matches an MXFP8 chain (``float8_e4m3fn``
    activations, ``float8_e8m0fnu`` scales) and rewrites it to
    ``mo.composite.mega_ffn_nvfp4``. Compiling all the way also exercises the
    registration's ``float8_e4m3fn`` branch -> ``mega_ffn_mxfp8_dispatch``, and
    ``clamp=True`` exercises the OpenAI-style clamped SwiGLU selector.
    """
    graph = build_graph_mxfp8(
        E, M, D, K1, N2, SF_DIM_0, DeviceRef.GPU(), DeviceRef.CPU(), clamp=True
    )
    _assert_fused(*_compile_ir(graph, tmp_path))
