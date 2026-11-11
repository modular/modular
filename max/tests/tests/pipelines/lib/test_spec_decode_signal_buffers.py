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
"""Whether a spec-decode graph takes signal buffers.

The answer fixes the graph's signature, so it must not depend on whether the
buffers were allocated: under virtual devices none are.
"""

from types import SimpleNamespace

import pytest
from max.driver import CPU
from max.nn.legacy.comm import Signals
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
)
from max.pipelines.speculative import spec_adapters


def _model(num_devices: int) -> SpecDecodePipelineModel:
    model = object.__new__(SpecDecodePipelineModel)
    model.devices = [CPU()] * num_devices
    model.__dict__["pairing"] = SimpleNamespace(
        target=SimpleNamespace(pipeline_model=None), adapter_arch="Target"
    )
    return model


@pytest.mark.parametrize(
    ("num_devices", "index", "expected"),
    [(1, None, False), (2, None, True), (1, 0, True)],
)
def test_graph_takes_signal_buffers_without_allocating_them(
    monkeypatch: pytest.MonkeyPatch,
    num_devices: int,
    index: int | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        spec_adapters, "signal_buffer_index", lambda arch: index
    )
    monkeypatch.setattr(Signals, "allocate", staticmethod(lambda devices: []))
    model = _model(num_devices)

    assert model.takes_signal_buffers is expected
    assert model.signal_buffers == []
