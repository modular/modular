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
"""The fused W4A8 MoE FFN op through a graph, against the two-launch chain.

Stages the op's graph and the chain's graph on MiMo-V2.6-Flash's expert slice and
executes both. The op picks its own regime from the routed rows -- one launch
inside ``BAND_ROWS`` and above ``FLOOR_ROWS``, the chain's own kernels outside
it -- so one compiled model covers both arms: a c1 batch's 64 rows are the
chain's arm and a c8 batch's 512 rows the one launch's, byte-for-byte against
the chain either way, with the bf16 result, the E4M3 intermediate and that
intermediate's scale tiles all compared. Each case also runs the op twice back
to back, and once more as a captured device graph replayed, on the one counter
workspace the graph hands it, so the generation parity that keeps the pool
counters correct across launches is covered where a model would exercise it.
The replays go further: the token rows are overwritten in place between replays
of the same key, so a replay that read its capture's inputs instead of the
buffers' current bytes fails here, and the first key is replayed again after a
second batch was captured and replayed under its own, so a key is held to its
own batch rather than to the newest one.

One test builds both graphs with a symbolic row dimension instead, the shape a
captured step has, and runs the one compiled model at five row counts: at the
floor and one step (8 rows) past it, at the ceiling and one step past it, and at
a prefill-shaped batch. Each gets its own zeroed counter workspace, so the
workspace says which arm ran -- only the one launch writes it -- and an
off-by-one at either bound fails here even when the bytes agree. A graph that
forced the rows static would not load against a captured batch, and one that
mishandled the dynamic tail would be wrong only where no test ran.

One more runs a sparse routing -- 8 of the 64 slots active and the rest carrying
-1 -- through the chain arm, which is the arm that reads the routing arrays
rather than re-deriving them.

The row estimate the two graphs launch with is not policed here: `usage_stats`
on the chain's side and `estimated_total_m` on the op's only select each
dispatcher's launch config (its MMA tile and CTA group from rows per active
expert), so a mismatched estimate costs speed rather than bytes, and the
byte compare above would not see it. `_megaffn_mimo_graphs.build_chain_graph`
passes the same routed row count the layer passes, which is what keeps the two
arms on the same config.

This is a graph-level test on purpose. The kernel's numerics are gated in Mojo
(`Kernels/test/gpu/Moka/test/test_megaffn_mimo_port.mojo` against the component,
`Kernels/test/gpu/linalg/test_mega_ffn_mimo_port.mojo` against the chain), and
neither can see what this one can: that the op's operands reach the kernel in the
order it reads them, that the counter workspace is threaded through the op's
mutation, and that a second execution and a replay read the current inputs. A
wrong operand order or a stale workspace is silent in the kernel tests and wrong
logits in a model, so it belongs before merge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from _megaffn_mimo_graphs import (
    BAND_ROWS,
    FLOOR_ROWS,
    MEGAFFN_MIMO_COUNTER_WORDS,
    build_chain_graph,
    build_port_graph,
    input_specs,
    np_inputs,
)
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType, Type
from max.nn._graph_ops import MEGAFFN_MIMO_GEN_WORD, MEGAFFN_MIMO_HWM_WORD

NUM_EXPERTS = 64
"""MiMo's slice is 256 experts; 64 keeps this test's weights near 200 MiB and
still walks a multi-slot expert space. The 256-expert path is gated in Mojo."""
NUM_ACTIVE = 64
"""Every expert carries rows here, the layout the chain's front-packed contract
wants; empty and `-1` slots are covered in the Mojo tests."""
REQUESTS = (8, 64)
"""The c1 and c8 decode batches: 64 and 512 routed rows at top-8 over these 64
experts, i.e. one row per expert and eight. The c1 batch is the op's chain arm
(64 rows is exactly `FLOOR_ROWS`, which is exclusive) and the c8 batch is its
one-launch arm, so one compiled model exercises both."""

Spec = list[tuple[str, Type[Any]]]

RUNS_PER_CASE = 2
"""Executions of each row count on its own workspace, which is one turn of the
rotation: the second launch counts into the other half of every pool word and
clears the half the first left, so it runs on counters its predecessor wrote
rather than on the fresh ones. Two is a turn, not a cycle -- the half a third
launch would return to is not read here."""


def _staged_buffer(
    arr: np.ndarray, dtype: DType, device: Accelerator, host: bool
) -> Buffer:
    """Copies `arr` into a `dtype` buffer, its bytes reinterpreted.

    `host` keeps the buffer on the CPU, for the operand a graph declares there.
    """
    buf = Buffer.from_dlpack(torch.from_numpy(np.ascontiguousarray(arr)))
    if dtype != DType.uint8 and arr.dtype == np.uint8:
        buf = buf.view(dtype)
    return buf if host else buf.to(device)


def _host(buf: Buffer) -> np.ndarray:
    """Returns a buffer's bytes as a host array: bit patterns, not values."""
    return buf.to(CPU()).view(DType.uint8).to_numpy()


def _dtype_of(specs: Spec, name: str) -> DType:
    for spec_name, spec_type in specs:
        if spec_name != name:
            continue
        if isinstance(spec_type, (TensorType, BufferType)):
            return spec_type.dtype
        raise TypeError(f"{name} is neither a tensor nor a buffer")
    raise KeyError(name)


def _stage(
    specs: Spec,
    arrays: dict[str, np.ndarray],
    device: Accelerator,
    counters: Buffer | None = None,
) -> list[Buffer]:
    """Stages one graph's inputs, in the order its signature declares them.

    `counters` reuses one call's persistent counter workspace across calls, which
    is how a model hands the same layer's buffer to every step.
    """
    buffers: list[Buffer] = []
    for name, spec_type in specs:
        if name == "arrival_count":
            # The persistent counter workspace: zeroed once, as a model would.
            buffers.append(
                counters
                if counters is not None
                else Buffer.zeros(
                    [MEGAFFN_MIMO_COUNTER_WORDS], DType.uint32, device
                )
            )
        else:
            host = (
                isinstance(spec_type, TensorType)
                and spec_type.device == DeviceRef.CPU()
            )
            buffers.append(
                _staged_buffer(
                    arrays[name], _dtype_of(specs, name), device, host
                )
            )
    return buffers


def _mutate_tokens(
    specs: Spec,
    inputs: list[Buffer],
    arrays: dict[str, np.ndarray],
    device: Accelerator,
) -> None:
    """Overwrites the staged graph's token rows with `arrays`, in place.

    In place on the same buffer, not a new one: a replay is tied to the input
    buffers it was captured with, and the point here is that it reads their
    current bytes rather than the ones it saw when it was captured.
    """
    index = [name for name, _ in specs].index("hidden_states")
    inputs[index].inplace_copy_from(
        _staged_buffer(
            arrays["hidden_states"],
            _dtype_of(specs, "hidden_states"),
            device,
            host=False,
        )
    )


def _retokenized(
    arrays: dict[str, np.ndarray],
    seed: int,
    rows_per_expert: int,
) -> dict[str, np.ndarray]:
    """Returns `arrays` with only the token rows replaced, drawn from `seed`.

    The weights, the scales and the routing stay the graph's: only the token
    rows are rewritten in place, so only they may differ from the reference.
    """
    fresh = np_inputs(NUM_EXPERTS, NUM_ACTIVE, rows_per_expert, seed=seed)
    return {**arrays, "hidden_states": fresh["hidden_states"]}


@pytest.mark.parametrize("requests", REQUESTS)
def test_op_matches_the_chain_through_a_graph(requests: int) -> None:
    rows_per_expert = requests * 8 // NUM_ACTIVE
    arrays = np_inputs(NUM_EXPERTS, NUM_ACTIVE, rows_per_expert)
    rows = int(arrays["hidden_states"].shape[0])
    arm = _arm(rows)
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    port_specs, chain_specs = input_specs(
        NUM_EXPERTS,
        arrays["hidden_states"].shape[0],
        arrays["a_scales"].shape[0],
        device_ref,
        DeviceRef.CPU(),
    )
    session = InferenceSession(devices=[device])
    port: Model = session.load(build_port_graph(port_specs))
    chain: Model = session.load(build_chain_graph(chain_specs))

    port_inputs = _stage(port_specs, arrays, device)
    expected = chain.execute(*_stage(chain_specs, arrays, device))

    def _check(got: list[Buffer], at: str) -> None:
        for name, index in (
            ("result", 0),
            ("intermediate", 1),
            ("scales", 2),
        ):
            np.testing.assert_array_equal(
                _host(got[index]),
                _host(expected[index]),
                err_msg=f"{at} {name}",
            )

    # Two executions on the one workspace, which is how a model hands the same
    # layer's buffer to every step: an in-band second launch counts into the
    # other half of every pool word, and a chain-arm one leaves the workspace
    # alone between them.
    for attempt in range(2):
        _check(
            port.execute(*port_inputs),
            f"requests={requests} {arm} run {attempt}",
        )

    # A captured device graph replays the same work on the same workspace, and
    # the replay must land the same bytes. Capture's own return is only filled
    # by a replay, so it is compared after one. The arm a capture records is the
    # one its batch takes, which is why the regime is decided where the rows
    # are rather than when the graph is built.
    keys = [requests]
    captured = port.capture(keys, *port_inputs)
    port.debug_verify_replay(keys, *port_inputs)
    port.replay(keys, *port_inputs)
    _check(captured, f"requests={requests} {arm} replay")

    # The capture records the buffers, not their contents, so a replay after the
    # tokens are overwritten in place must follow them. Otherwise a model would
    # keep serving its first batch's logits, which no byte compare of a single
    # replay can see.
    retoken = _retokenized(arrays, 11, rows_per_expert)
    _mutate_tokens(port_specs, port_inputs, retoken, device)
    port.replay(keys, *port_inputs)
    retoken_expected = chain.execute(*_stage(chain_specs, retoken, device))
    for name, index in (("result", 0), ("intermediate", 1), ("scales", 2)):
        np.testing.assert_array_equal(
            _host(captured[index]),
            _host(retoken_expected[index]),
            err_msg=f"requests={requests} replay of rewritten tokens {name}",
        )

    # A second batch captures and replays under its own key (the record/replay
    # API ties a key to its input signature), and must land that batch's bytes.
    other = np_inputs(NUM_EXPERTS, NUM_ACTIVE, rows_per_expert, seed=7)
    other_expected = chain.execute(*_stage(chain_specs, other, device))
    other_keys = [requests + 1000]
    other_inputs = _stage(port_specs, other, device)
    other_captured = port.capture(other_keys, *other_inputs)
    port.replay(other_keys, *other_inputs)
    for name, index in (("result", 0), ("intermediate", 1), ("scales", 2)):
        np.testing.assert_array_equal(
            _host(other_captured[index]),
            _host(other_expected[index]),
            err_msg=f"requests={requests} replay of a second batch {name}",
        )

    # And the first key is still live afterwards: replaying it again, with its
    # input buffers rewritten once more, must land that content, not the
    # second batch's and not its own previous batch's.
    again = _retokenized(arrays, 13, rows_per_expert)
    _mutate_tokens(port_specs, port_inputs, again, device)
    port.replay(keys, *port_inputs)
    again_expected = chain.execute(*_stage(chain_specs, again, device))
    for name, index in (("result", 0), ("intermediate", 1), ("scales", 2)):
        np.testing.assert_array_equal(
            _host(captured[index]),
            _host(again_expected[index]),
            err_msg=f"requests={requests} replay of key A after key B {name}",
        )


ROWS_CASES = {
    "floor": (8, 8),
    "floor+8": (9, 8),
    "band": (64, 8),
    "band+8": (52, 10),
    "prefill": (64, 16),
}
"""The row counts one compiled model must carry, each with its active experts
and rows per expert: 8 experts holding 8 rows (64 routed rows, exactly the
floor, which is exclusive, so the op runs the chain), 9 holding 8 (72, the
first count past the floor, so the op runs the one launch), 64 holding 8 (512,
the ceiling itself, so the one launch), 52 holding 10 (520, the first count
past the ceiling, so the chain), and 64 holding 16 (1024, a prefill-shaped
batch where each expert holds two pools of rows). Both of the op's arms ride
one compile, and both bounds are straddled by one step of 8 rows on each side,
so an off-by-one in either comparison changes which arm runs and shows up in the
counter workspace the one launch counts into."""


def _arm(rows: int) -> str:
    """Names the arm the op takes at `rows`, for a failure message."""
    if not FLOOR_ROWS < rows <= BAND_ROWS:
        return f"chain arm ({rows} rows outside ({FLOOR_ROWS}, {BAND_ROWS}])"
    return f"one-launch arm ({rows} rows in ({FLOOR_ROWS}, {BAND_ROWS}])"


def _pools(num_active_experts: int, rows_per_expert: int) -> int:
    """The pool counters a launch over that routing counts into.

    A pool is 8 token rows of one expert, so an expert's rows take
    `ceildiv(rows_per_expert, 8)` of them.
    """
    return num_active_experts * ((rows_per_expert + 7) // 8)


def _assert_arm_ran(
    counters: Buffer,
    num_active_experts: int,
    rows_per_expert: int,
    launches: int,
) -> None:
    """Checks the counter workspace after `launches` calls on it.

    The workspace is what separates the two arms from the outside: only the one
    launch touches it -- the generation it advances is what selects the half of
    every pool word its counts go in, and the high-water mark it records is the
    pool count its walk read -- while the chain it runs out of band never looks
    at it. So an arm that ran says so here even when its bytes agree, which is
    what an off-by-one at either bound would otherwise hide behind.
    """
    words = counters.to(CPU()).view(DType.uint32).to_numpy()
    rows = num_active_experts * rows_per_expert
    if not FLOOR_ROWS < rows <= BAND_ROWS:
        assert not words.any(), (
            f"{rows} rows should run the chain, which leaves the counters"
            " alone, but they hold"
            f" {words[words != 0][:8]}"
        )
        return
    assert words[MEGAFFN_MIMO_GEN_WORD] == launches, (
        f"{rows} rows should run the one launch {launches} times on this"
        f" workspace, so its generation is {launches}; the words are"
        f" {words[:32]}"
    )
    assert words[MEGAFFN_MIMO_HWM_WORD] == _pools(
        num_active_experts, rows_per_expert
    ), (
        f"{rows} rows should count into {_pools(num_active_experts, rows_per_expert)}"
        f" pools; the words are {words[:32]}"
    )


def test_one_compiled_op_runs_every_row_count() -> None:
    """A symbolic row dimension, one compile, both arms, byte-exact.

    MiMo's routed rows are ``topk_ids.shape[0]``, which is symbolic under device
    graph capture, so the op has to carry a dynamic M: a graph whose row
    dimension is static would fail to load against the captured batch, and one
    that is dynamic but mishandles the tail would be wrong only at the row
    counts no test ran. Both graphs here -- the op's and the chain's -- are built
    with a symbolic row dimension and a symbolic scale-row dimension, compiled
    once, and then executed at each row count. The counts straddle both bounds by
    one step of 8 rows, so the op's own regime choice is executed on both of its
    arms against the chain, and the counter workspace says which arm ran.

    Each case gets its own zeroed workspace, so "the counters are untouched"
    means the chain ran and "the generation advanced" means the one launch did,
    rather than one case's counts being read as the next case's.
    """
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    port_specs, chain_specs = input_specs(
        NUM_EXPERTS, "rows", "scale_rows", device_ref, DeviceRef.CPU()
    )
    session = InferenceSession(devices=[device])
    port: Model = session.load(build_port_graph(port_specs))
    chain: Model = session.load(build_chain_graph(chain_specs))

    for name, (active, per_expert) in ROWS_CASES.items():
        arrays = np_inputs(NUM_EXPERTS, active, per_expert)
        rows = active * per_expert
        assert arrays["hidden_states"].shape[0] == rows
        expected = chain.execute(*_stage(chain_specs, arrays, device))
        counters = Buffer.zeros(
            [MEGAFFN_MIMO_COUNTER_WORDS], DType.uint32, device
        )
        for attempt in range(RUNS_PER_CASE):
            got = port.execute(
                *_stage(port_specs, arrays, device, counters=counters)
            )
            for output, index in (
                ("result", 0),
                ("intermediate", 1),
                ("scales", 2),
            ):
                np.testing.assert_array_equal(
                    _host(got[index]),
                    _host(expected[index]),
                    err_msg=(
                        f"{name} ({rows} rows), {_arm(rows)},"
                        f" run {attempt} {output}"
                    ),
                )
        _assert_arm_ran(counters, active, per_expert, launches=RUNS_PER_CASE)


SPARSE = (8, 8)
"""8 experts of the 64 slots with 8 rows each -- 64 routed rows, the floor, so
the op runs its chain arm -- spread one every 8 slots, with -1 in the 56 slots
between them."""


def test_a_sparse_routing_matches_the_chain() -> None:
    """Empty expert slots and -1 ids, on the arm that walks the caller's routing.

    A routing's slots are not always packed: an expert can hold no tokens while
    the slot after it holds many, and a slot can carry no weights at all. The
    op's chain arm passes the caller's routing arrays to the same two dispatches
    the graph's grouped matmul ops do, so a layout has to mean the same thing on
    both sides -- including the id a zero-width slot carries, which both
    schedulers skip before any load. The 64 rows put this on the chain arm,
    which is the arm that reads the routing rather than re-deriving it from the
    counters.
    """
    active, per_expert = SPARSE
    arrays = np_inputs(NUM_EXPERTS, active, per_expert, sparse=True)
    rows = int(arrays["hidden_states"].shape[0])
    assert rows == active * per_expert
    assert _arm(rows).startswith("chain arm"), _arm(rows)
    # The layout the test is for: every other slot empty, and empty slots told
    # apart by a negative id rather than by a range the walk has to measure.
    assert int((arrays["expert_ids"] < 0).sum()) == NUM_EXPERTS - active
    active_slots = np.flatnonzero(np.diff(arrays["expert_start_indices"]))
    assert active_slots.tolist() == list(
        range(0, NUM_EXPERTS, NUM_EXPERTS // active)
    ), active_slots

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    port_specs, chain_specs = input_specs(
        NUM_EXPERTS,
        arrays["hidden_states"].shape[0],
        arrays["a_scales"].shape[0],
        device_ref,
        DeviceRef.CPU(),
    )
    session = InferenceSession(devices=[device])
    port: Model = session.load(build_port_graph(port_specs))
    chain: Model = session.load(build_chain_graph(chain_specs))
    counters = Buffer.zeros([MEGAFFN_MIMO_COUNTER_WORDS], DType.uint32, device)
    expected = chain.execute(*_stage(chain_specs, arrays, device))
    got = port.execute(*_stage(port_specs, arrays, device, counters=counters))
    for name, index in (("result", 0), ("intermediate", 1), ("scales", 2)):
        np.testing.assert_array_equal(
            _host(got[index]),
            _host(expected[index]),
            err_msg=f"sparse routing {name}",
        )
    _assert_arm_ran(counters, active, per_expert, launches=1)
