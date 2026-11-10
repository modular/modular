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

"""Gated DeltaNet recurrence routing, read off the built graph.

The kernel-level differential test grades ``kda_chunk_launch`` on tensors it
builds itself, which says nothing about what this layer hands the op. What can
go wrong here is wiring rather than arithmetic, and it goes wrong quietly: a
gate spread that lowers to a view still produces the right *values* -- every
channel of a head holds the same scalar -- so a shape-only or output-only check
passes while the descriptor reads past its buffer.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops.state_space import verify_width_operand
from max.nn.legacy import Module
from max.pipelines.architectures.qwen3_5.layers import gated_deltanet
from max.pipelines.architectures.qwen3_5.layers.gated_deltanet import (
    GatedDeltaNet,
)
from max.pipelines.architectures.qwen3_5.state_cache import (
    RING_LEAF_ID,
    GatedDeltaRingAccess,
    linear_state_regions,
    ring_len_for_window,
)

DEVICE = DeviceRef.GPU(0)
HIDDEN = 5120
NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
HEAD_DIM = 128
CONV_KERNEL = 4
SLOTS = 4
NUM_DRAFTS = 3
RING_LEN = 4


def _qualify_weight_names(layer: Module, prefix: str = "gdn") -> None:
    """Give every weight its module path, the way a parent module would.

    Built standalone, the layer holds several weights named plainly "weight"
    (the norm's and the projections'), and the graph refuses a duplicate. In a
    real model the enclosing modules supply the path.
    """
    for name, weight in layer.layer_weights.items():
        weight.name = f"{prefix}.{name}"
    for name, sub in layer.sublayers.items():
        _qualify_weight_names(sub, f"{prefix}.{name}")


def _forward_mlir(monkeypatch: pytest.MonkeyPatch, architecture: str) -> str:
    """Builds the layer as if on an NVIDIA GPU of ``architecture``."""
    monkeypatch.setattr(gated_deltanet, "accelerator_api", lambda: "cuda")
    monkeypatch.setattr(
        gated_deltanet, "accelerator_architecture_name", lambda: architecture
    )
    layer = GatedDeltaNet(
        hidden_size=HIDDEN,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        dtype=DType.bfloat16,
        device=DEVICE,
    )
    _qualify_weight_names(layer)
    conv_dim = NUM_KEY_HEADS * HEAD_DIM * 2 + NUM_VALUE_HEADS * HEAD_DIM
    with Graph(
        "gdn_routing",
        input_types=[
            TensorType(DType.bfloat16, ["total_tokens", HIDDEN], device=DEVICE),
            BufferType(
                DType.float32, [SLOTS, conv_dim, CONV_KERNEL - 1], device=DEVICE
            ),
            TensorType(DType.uint32, ["batch"], device=DEVICE),
            BufferType(
                DType.float32,
                [SLOTS, NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM],
                device=DEVICE,
            ),
            TensorType(DType.uint32, ["batch"], device=DEVICE),
            TensorType(DType.uint32, ["batch_plus_one"], device=DEVICE),
        ],
    ) as graph:
        x, conv_pool, conv_row, rec_pool, rec_row, offsets = graph.inputs
        out = layer(
            x.tensor,
            conv_pool.buffer,
            conv_row.tensor,
            rec_pool.buffer,
            rec_row.tensor,
            offsets.tensor,
        )
        graph.output(out)
    return str(graph)


# The graph is only built, never compiled or run, so the routing is checked
# for both sides of the SM100 gate from any host, with the driver queries
# patched to report the architecture under test.
PRE_SM100 = ["sm_86", "sm_90a"]
SM100_PLUS = ["sm_100a", "sm_103"]


@pytest.mark.parametrize("architecture", PRE_SM100)
def test_pre_sm100_prefill_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """The fused chunk kernel doesn't instantiate below SM100 (A10G, H100)."""
    mlir = _forward_mlir(monkeypatch, architecture)
    assert "kda_chunk" not in mlir
    assert "gated_delta_recurrence" in mlir


@pytest.mark.parametrize(
    ("architecture", "capability"),
    [
        ("sm_86", 86),
        ("sm_90a", 90),
        ("sm_100a", 100),
        ("sm_103", 103),
        ("gfx942", 0),
        ("", 0),
    ],
)
def test_compute_capability_is_read_from_the_architecture_name(
    monkeypatch: pytest.MonkeyPatch, architecture: str, capability: int
) -> None:
    monkeypatch.setattr(
        gated_deltanet, "accelerator_architecture_name", lambda: architecture
    )
    assert gated_deltanet._nvidia_compute_capability() == capability


@pytest.mark.parametrize("architecture", SM100_PLUS)
def test_prefill_and_decode_are_both_wired(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """Both recurrences are present, behind a conditional."""
    mlir = _forward_mlir(monkeypatch, architecture)
    assert "kda_chunk" in mlir, "prefill is not routed to the fused chunk op"
    assert "gated_delta_recurrence" in mlir, (
        "decode must stay on the sequential recurrence"
    )
    assert "mo.if" in mlir, (
        "the two recurrences must sit behind a conditional, not run in series"
    )


def test_gate_spread_is_materialized_not_a_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate spread must not lower to a broadcast view.

    ``mo.broadcast_to`` carries ``MO_ViewLike``, so its result may keep a
    stride-0 buffer holding one column per head. The chunk op reads the gate
    through a TMA descriptor over a flat ``[token, HV*K]`` shape and would walk
    ``K`` times past the end of such a buffer. An elementwise multiply is not
    view-like and allocates the whole thing, so the gate is spread that way.
    """
    mlir = _forward_mlir(monkeypatch, "sm_100a")
    kda_chunk_lines = [
        line for line in mlir.splitlines() if "kda_chunk" in line
    ]
    assert kda_chunk_lines, "no kda_chunk op in the graph to check"
    assert "mo.broadcast_to" not in mlir, (
        "the gate spread regressed to a broadcast view; a TMA descriptor over "
        "the flat gate would read past a buffer 1/K the size"
    )


def _verify_mlir(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    num_drafts: int = NUM_DRAFTS,
    ring_len: int = RING_LEN,
    head_dim: int = HEAD_DIM,
) -> str:
    """Builds a verify of the layer on ``architecture``."""
    monkeypatch.setattr(gated_deltanet, "accelerator_api", lambda: "cuda")
    monkeypatch.setattr(
        gated_deltanet, "accelerator_architecture_name", lambda: architecture
    )
    layer = GatedDeltaNet(
        hidden_size=HIDDEN,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
        conv_kernel_size=CONV_KERNEL,
        dtype=DType.bfloat16,
        device=DEVICE,
    )
    _qualify_weight_names(layer)
    conv_dim = NUM_KEY_HEADS * head_dim * 2 + NUM_VALUE_HEADS * head_dim
    (ring,) = (
        region
        for region in linear_state_regions(
            num_linear_layers=1,
            key_head_dim=head_dim,
            num_key_heads=NUM_KEY_HEADS,
            value_head_dim=head_dim,
            num_value_heads=NUM_VALUE_HEADS,
            conv_kernel_dim=CONV_KERNEL,
            dtype=DType.float32,
            num_devices=1,
            ring_len=ring_len,
        )
        if region.leaf_id == RING_LEAF_ID
    )
    rows = TensorType(DType.uint32, ["batch_size"], device=DEVICE)
    with Graph(
        "gdn_verify_routing",
        input_types=[
            TensorType(DType.bfloat16, ["total_tokens", HIDDEN], device=DEVICE),
            BufferType(
                DType.float32, [SLOTS, conv_dim, CONV_KERNEL - 1], device=DEVICE
            ),
            rows,
            BufferType(
                DType.float32,
                [SLOTS, NUM_VALUE_HEADS, head_dim, head_dim],
                device=DEVICE,
            ),
            rows,
            TensorType(DType.uint32, ["offsets_len"], device=DEVICE),
            BufferType(DType.float32, [SLOTS, *ring.row_shape], device=DEVICE),
            rows,
        ],
    ) as graph:
        (
            x,
            conv_pool,
            conv_row,
            rec_pool,
            rec_row,
            offsets,
            pool,
            ring_row,
        ) = graph.inputs
        out = layer(
            x.tensor,
            conv_pool.buffer,
            conv_row.tensor,
            rec_pool.buffer,
            rec_row.tensor,
            offsets.tensor,
            ring=GatedDeltaRingAccess(pool=pool.buffer, row_id=ring_row.tensor),
            verify_width=verify_width_operand(num_drafts),
        )
        graph.output(out)
    return str(graph)


@pytest.mark.parametrize("architecture", SM100_PLUS)
def test_a_verify_runs_its_long_rows_on_the_chunk_op(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """The ring skips the long rows, which the chunk op runs instead."""
    mlir = _verify_mlir(monkeypatch, architecture)
    assert "kda_chunk_rows" in mlir
    op_lines = [l for l in mlir.splitlines() if "kda_chunk_rows" in l]
    # The op reads K from verify_width, so a missing parameter cannot
    # silently disable row select.
    assert "ring_len" not in op_lines[0]
    # The pool's layout reaches the op, so a V_FIRST pool gets a V_FIRST
    # kernel.
    assert "state_layout" in op_lines[0]
    assert "skip_long_rows = true" in mlir
    # Decode and mixed steps share one ring launch, so no third branch's
    # output has to be joined in every layer.
    assert "skip_long_rows = false" not in mlir
    # The long-row op sits inside the mixed-case cond (indented deeper than
    # the mo.if lines), and the plan's gather/scatter is gone.
    op_indent = min(len(l) - len(l.lstrip()) for l in op_lines)
    if_indent = max(
        len(l) - len(l.lstrip()) for l in mlir.splitlines() if "mo.if" in l
    )
    assert op_indent > if_indent
    assert "mo.gather" not in mlir
    assert "mo.scatter_nd" not in mlir


@pytest.mark.parametrize("architecture", PRE_SM100)
def test_a_verify_without_the_chunk_op_commits_its_long_rows(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """Below SM100 the ring's commit variant keeps the long rows."""
    mlir = _verify_mlir(monkeypatch, architecture)
    assert "kda_chunk" not in mlir
    assert "skip_long_rows = true" not in mlir
    assert "gated_delta_recurrence_verify_ring" in mlir


def test_an_unfused_geometry_keeps_the_serial_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head width the fused arm rejects runs the non-skip ring."""
    mlir = _verify_mlir(monkeypatch, "sm_100a", head_dim=32)
    assert "kda_chunk_rows" not in mlir
    assert "skip_long_rows = true" not in mlir
    assert "gated_delta_recurrence_verify_ring" in mlir
    assert "kda_chunk" in mlir


def _mixed_case_predicate(num_drafts: int, total_tokens: int) -> bool:
    """Evaluates the layer's mixed-case predicate for a batch of two rows."""
    with Graph("mixed_case_predicate", input_types=[]) as graph:
        total = ops.shape_to_tensor([total_tokens])[0]
        batch = ops.shape_to_tensor([3])[0] - 1
        prefill_case = gated_deltanet._prefill_case(
            verify_width_operand(num_drafts), total > batch
        )
        graph.output(
            gated_deltanet._mixed_case_predicate(
                verify_width_operand(num_drafts), prefill_case, total, batch
            )
        )
    return bool(InferenceSession().load(graph).execute()[0].to_numpy())


def test_a_decode_only_verify_takes_no_long_row_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every row is K + 1 tokens, so the mixed-case predicate is false."""
    assert not _mixed_case_predicate(NUM_DRAFTS, 2 * (NUM_DRAFTS + 1))
    mlir = _verify_mlir(monkeypatch, "sm_100a")
    assert "kda_chunk_rows" in mlir


def test_a_zero_width_prompt_takes_no_long_row_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At K = 0 the prefill case owns the step, not the long-row branch."""
    assert not _mixed_case_predicate(0, 8)
    mlir = _verify_mlir(monkeypatch, "sm_100a", num_drafts=0)
    assert "kda_chunk_rows" in mlir


@pytest.mark.parametrize(
    "num_drafts",
    [
        pytest.param(1, id="one-draft"),
        pytest.param(2, id="two-drafts"),
        pytest.param(3, id="three-drafts"),
        pytest.param(5, id="five-drafts"),
        pytest.param(7, id="seven-drafts"),
    ],
)
def test_the_conv_window_is_one_verify_window_per_row(
    monkeypatch: pytest.MonkeyPatch, num_drafts: int
) -> None:
    """The verify's conv copy holds one K + 1 slot per row, as on main."""
    mlir = _verify_mlir(
        monkeypatch,
        "sm_100a",
        num_drafts=num_drafts,
        ring_len=ring_len_for_window(num_drafts + 1),
    )
    window_lines = [
        line
        for line in mlir.splitlines()
        if "gated_delta_conv1d_verify_fwd" in line
    ]
    assert window_lines, "no conv verify op in the graph to check"
    # The op's window result type is [batch * (K + 1), conv_dim].
    assert any(
        f"mul_no_wrap(batch_size, {num_drafts + 1})" in line
        for line in window_lines
    ), "the conv window is not one verify window per row"


def test_a_narrower_step_keeps_its_own_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step narrower than the export's K still sizes its own window."""
    mlir = _verify_mlir(
        monkeypatch, "sm_100a", num_drafts=2, ring_len=ring_len_for_window(8)
    )
    window_lines = [
        line
        for line in mlir.splitlines()
        if "gated_delta_conv1d_verify_fwd" in line
    ]
    assert window_lines, "no conv verify op in the graph to check"
    assert any("mul_no_wrap(batch_size, 3)" in line for line in window_lines)
