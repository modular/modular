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

"""What a load hands back under virtual devices, and how it refuses.

Virtual-device mode latches process-wide at the first device creation, so this
needs a target of its own: the knobs are set at import, before anything below
reaches a device. The graphs are plain CPU graphs -- the stand-in is about
whether the process can execute at all, not about which arch it compiled for.

Eager tensors are out of scope: realizing one under virtual devices cannot work
at all, since the eager interpreter falls back to an artifact it loads by path
and ``init_all`` cannot name the graphs in one. That predates the stand-in.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

# A target that is deliberately not attached: virtual-device mode is the state
# where compiling is all a process can do.
set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_90a")
set_virtual_device_count(1)

import pytest
from max.driver import CPU, is_virtual_device_mode
from max.dtype import DType
from max.engine import (
    CompileOnlyExecutionError,
    CompileOnlyModel,
    InferenceSession,
    MefStore,
    Model,
)
from max.graph import DeviceRef, Graph, TensorType


def _graph(name: str = "add_one") -> Graph:
    dtype = TensorType(DType.float32, [4], device=DeviceRef.CPU())
    with Graph(name, input_types=[dtype]) as graph:
        graph.output(graph.inputs[0].tensor + 1.0)
    return graph


@pytest.fixture
def stand_in() -> CompileOnlyModel:
    model = InferenceSession(devices=[CPU()]).load(_graph())
    assert isinstance(model, CompileOnlyModel)
    return model


def test_virtual_mode_is_active() -> None:
    assert is_virtual_device_mode()


def test_a_load_completes_and_keeps_what_it_compiled(
    stand_in: CompileOnlyModel,
) -> None:
    # Completing the load is the point: a caller cross-compiling or measuring a
    # compile has no device to initialize on and still wants the call to return.
    assert stand_in.name == "add_one"
    assert stand_in.compiled is not None


@pytest.mark.parametrize(
    "call",
    [
        lambda model: model.execute(),
        lambda model: model(),
        lambda model: model.capture(0),
        lambda model: model.replay(0),
    ],
)
def test_executing_refuses(stand_in: CompileOnlyModel, call: object) -> None:
    # A stand-in that answered any of these with a placeholder would move the
    # failure to whatever consumed it, which is how a compile-only run used to
    # surface as a fault inside a kernel.
    assert callable(call)
    with pytest.raises(CompileOnlyExecutionError):
        call(stand_in)


# What the stand-in answers rather than refuses: both are known without a
# device, and `execute` refuses when called rather than when looked up.
_CARRIED = frozenset({"name", "compiled", "execute"})


@pytest.mark.parametrize(
    "member",
    sorted(
        member
        for member in dir(Model)
        if not member.startswith("__") and member not in _CARRIED
    ),
)
def test_every_model_member_refuses(
    stand_in: CompileOnlyModel, member: str
) -> None:
    # Derived from Model rather than listed, because a load hands this back
    # where a Model is expected: a member left out answers with an
    # AttributeError, which reads as a broken caller rather than as the
    # absence of the device, and one added to Model later is covered here
    # without anyone remembering to.
    with pytest.raises(CompileOnlyExecutionError):
        getattr(stand_in, member)


def test_an_attribute_no_model_has_stays_an_attribute_error(
    stand_in: CompileOnlyModel,
) -> None:
    # The refusal covers Model's surface, not every typo.
    missing = "not_a_member_of_any_model"
    with pytest.raises(AttributeError):
        getattr(stand_in, missing)


@pytest.fixture
def export_store(tmp_path: Path) -> Iterator[MefStore]:
    store = MefStore.for_export(tmp_path)
    previous = InferenceSession.default_mef_store
    InferenceSession.default_mef_store = store
    try:
        yield store
    finally:
        InferenceSession.default_mef_store = previous


def test_recording_a_graph_survives_the_refusal(
    export_store: MefStore, tmp_path: Path
) -> None:
    # The shape of a record run: the artifact is written by the compile half,
    # which finished, and only what the device would have done is refused.
    model = InferenceSession(devices=[CPU()]).load(_graph("recorded"))
    export_store.write_manifest()

    assert len(list(tmp_path.glob("*.mef"))) == 1
    with pytest.raises(CompileOnlyExecutionError):
        model.execute()
