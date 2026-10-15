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
"""Is ``keyed_uniform`` a sound accept coin through the graph API?

The sampled verdict accepts a draft with probability ``min(1, p_target /
q_draft)``, from a coin that must be keyed per request: ``ops.random.uniform``
reads index 0 of the graph seed, so a request's draw there rides on whatever
else shares the batch. ``keyed_uniform`` gives every row its own Philox key.

``max/kernels/test/gpu/nn/test_keyed_uniform.mojo`` pins the kernel itself.
This pins the op as the sampler reaches it -- that the draw is uniform and
keyed per row through the graph, and that ``coin * q >= p`` realizes the
acceptance probability the rejection sampler is owed, saturating cases
included.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import keyed_uniform

# Rows per setting: 1/sqrt(16384) puts the 4-sigma band near +/-1.6pp at
# p=0.5, tight enough that a coin off its analytic rate cannot hide in it.
ROWS = 16384

# (p_target, q_draft) with p <= q, so the analytic accept rate is p/q and the
# draw is genuinely random rather than saturated.
PQ_PAIRS = (
    (0.1, 0.9),
    (0.25, 0.5),
    (0.4, 0.8),
    (0.45, 0.5),
    (0.05, 0.95),
)


@pytest.fixture(scope="module")
def coin(session: InferenceSession) -> Model:
    """The verdict's coin, as ``_sampled_draft_verdict`` expresses it."""
    device_ref = DeviceRef.from_device(session.devices[0])
    row_type = TensorType(DType.float32, ["rows"], device=device_ref)
    seed_type = TensorType(DType.uint64, ["rows"], device=device_ref)
    with Graph(
        "keyed_uniform_coin", input_types=(row_type, row_type, seed_type)
    ) as graph:
        p_target, q_eff, seed = (value.tensor for value in graph.inputs)
        coins = keyed_uniform(seed)
        rejected = coins * q_eff >= p_target
        graph.output(rejected.cast(DType.int64), coins)
    return session.load(graph)


def _draw(
    coin: Model,
    session: InferenceSession,
    p_target: npt.NDArray[np.float32],
    q_eff: npt.NDArray[np.float32],
    seeds: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Returns the per-row rejection mask and the uniform draws behind it."""
    device = session.devices[0]
    rejected, coins = coin.execute(
        Buffer.from_numpy(p_target).to(device),
        Buffer.from_numpy(q_eff).to(device),
        Buffer.from_numpy(seeds).to(device),
    )
    assert isinstance(rejected, Buffer)
    assert isinstance(coins, Buffer)
    return rejected.to_numpy(), coins.to_numpy()


def test_draw_is_uniform(coin: Model, session: InferenceSession) -> None:
    """The marginal over distinct seeds is uniform on ``[0, 1)``."""
    seeds = np.arange(ROWS, dtype=np.uint64) + np.uint64(0x51EDC0DE)
    ones = np.ones(ROWS, dtype=np.float32)
    _, coins = _draw(coin, session, ones, ones, seeds)

    assert coins.min() >= 0.0 and coins.max() < 1.0, (
        f"draws left [0, 1): [{coins.min()}, {coins.max()}]"
    )

    buckets = 16
    counts = np.bincount((coins * buckets).astype(np.int64), minlength=buckets)
    expected = ROWS / buckets
    # 4 sigma on each bucket's binomial count.
    band = 4.0 * float(np.sqrt(expected * (1.0 - 1.0 / buckets)))
    worst = int(np.abs(counts - expected).max())
    assert worst < band, (
        f"bucket occupancy is off by {worst} against {expected:.0f} "
        f"(4-sigma band {band:.1f}); the draw is not uniform"
    )


def test_equal_seeds_agree_and_row_position_does_not_matter(
    coin: Model, session: InferenceSession
) -> None:
    """A row's draw is a function of its seed alone, not of where it sits.

    This is what ``ops.random.uniform`` cannot give: it walks the flat
    element index as its Philox counter, so a value there moves with the
    row. A request's coin has to survive being rebatched beside different
    co-residents.
    """
    ones = np.ones(ROWS, dtype=np.float32)
    probe = np.uint64(0xA5A51234DEADBEEF)

    shared = np.full(ROWS, probe, dtype=np.uint64)
    _, repeated = _draw(coin, session, ones, ones, shared)
    assert len(np.unique(repeated)) == 1, (
        "rows sharing one seed drew different values; the draw is not a "
        "function of the seed alone"
    )

    distinct = np.arange(ROWS, dtype=np.uint64)
    distinct[0] = probe
    distinct[ROWS // 3] = probe
    distinct[ROWS - 1] = probe
    _, mixed = _draw(coin, session, ones, ones, distinct)
    for row in (0, ROWS // 3, ROWS - 1):
        assert mixed[row] == repeated[0], (
            f"row {row} drew {mixed[row]} for a seed that drew "
            f"{repeated[0]} elsewhere; the draw reads its row index"
        )
    assert len(np.unique(mixed)) > ROWS // 2, (
        "distinct seeds collapsed onto a handful of values; rows are not "
        "decorrelating"
    )


@pytest.mark.parametrize(("p", "q"), PQ_PAIRS)
def test_accept_rate_matches_min_one_p_over_q(
    coin: Model, session: InferenceSession, p: float, q: float
) -> None:
    """``coin * q >= p`` rejects at exactly ``1 - min(1, p/q)``."""
    p_target = np.full(ROWS, p, dtype=np.float32)
    q_eff = np.full(ROWS, q, dtype=np.float32)
    seeds = np.arange(ROWS, dtype=np.uint64)

    rejected, _ = _draw(coin, session, p_target, q_eff, seeds)
    realized = float((rejected == 0).mean())
    expected = min(1.0, p / q)
    # 4 sigma on a binomial proportion; ROWS draws, independent by seed.
    band = 4.0 * float(np.sqrt(expected * (1.0 - expected) / ROWS))

    assert abs(realized - expected) < band, (
        f"accept rate {realized:.4f} != analytic {expected:.4f} "
        f"(4-sigma band {band:.4f}) for p={p}, q={q}"
    )


def test_target_mass_at_least_q_always_accepts(
    coin: Model, session: InferenceSession
) -> None:
    """``p >= q`` accepts unconditionally, because the draw is below 1."""
    q_eff = np.full(ROWS, 0.3, dtype=np.float32)
    rejected, _ = _draw(
        coin,
        session,
        q_eff.copy(),
        q_eff,
        np.arange(ROWS, dtype=np.uint64),
    )
    assert np.all(rejected == 0), (
        f"{int((rejected != 0).sum())} rows rejected a draft the target "
        "gives at least as much mass as the proposal did"
    )


def test_zero_target_mass_always_rejects(
    coin: Model, session: InferenceSession
) -> None:
    """A target that gives the drafted token no mass can never accept it.

    This is the orientation that makes a ``q`` underflowing to 0 degrade
    toward reject: the test multiplies rather than divides, so it never
    produces an infinity to compare against.
    """
    rejected, _ = _draw(
        coin,
        session,
        np.zeros(ROWS, dtype=np.float32),
        np.full(ROWS, 0.7, dtype=np.float32),
        np.arange(ROWS, dtype=np.uint64),
    )
    assert np.all(rejected == 1), (
        f"{int((rejected == 0).sum())} rows accepted a token the target "
        "gives no mass"
    )


def test_cpu_draws_the_same_values_as_the_gpu(
    session: InferenceSession,
) -> None:
    """The op is target-generic, and both targets draw bit-identical values.

    The kernel goes through ``elementwise``, so it builds for either target
    off the same Philox call. Agreement to the bit is the sharpest statement
    of what the draw is keyed on: nothing but the seed reaches it, not the
    device, the launch geometry, or the row.
    """
    seeds = np.array([0, 1, 42, 0xDEADBEEF, 999], dtype=np.uint64)

    def draw_on(device: Device) -> npt.NDArray[np.float32]:
        device_session = InferenceSession(devices=[device])
        device_ref = DeviceRef.from_device(device)
        seed_type = TensorType(DType.uint64, ["rows"], device=device_ref)
        with Graph("keyed_uniform_only", input_types=(seed_type,)) as graph:
            graph.output(keyed_uniform(graph.inputs[0].tensor))
        drawn = device_session.load(graph).execute(
            Buffer.from_numpy(seeds).to(device)
        )[0]
        assert isinstance(drawn, Buffer)
        return drawn.to_numpy()

    on_cpu = draw_on(CPU())
    on_gpu = draw_on(session.devices[0])
    assert np.array_equal(on_cpu, on_gpu), (
        f"CPU drew {on_cpu} where the GPU drew {on_gpu}"
    )
