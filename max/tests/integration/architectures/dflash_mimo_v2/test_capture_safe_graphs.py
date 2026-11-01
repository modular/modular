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
"""Tests that the fused MiMo-V2 DFlash graph stays capturable across devices.

Device graph capture records one graph per device stream, and a GPU-to-GPU
transfer makes one stream wait on another's, which CUDA refuses to capture.
Every activation that crosses devices goes through a collective over the
signal buffers instead. The graphs are built on virtual B200s, never
compiled, so the test needs no GPU.
"""

from __future__ import annotations

import re

import pytest
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.graph import DeviceRef, Graph
from mimo_dflash_harness import fused_graph, tiny_model
from tiny_mimo_dflash import Model

DEVICES = 2
GPU_TO_GPU = re.compile(r"mo\.transfer\[[^\n]*, gpu:\d+> to <\"gpu\", \d+>")


@pytest.fixture(scope="module")
def model() -> Model:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(DEVICES)
    return tiny_model()


def _refs() -> list[DeviceRef]:
    return [DeviceRef.GPU(i) for i in range(DEVICES)]


def _assert_capturable(graph: Graph) -> None:
    ir = str(graph)
    assert not GPU_TO_GPU.findall(ir), GPU_TO_GPU.findall(ir)
    assert re.search(r"mo\.distributed\.broadcast", ir)


@pytest.mark.parametrize("k", [3, 7])
def test_the_fused_spec_graph_crosses_devices_only_in_collectives(
    model: Model, k: int
) -> None:
    _assert_capturable(fused_graph(model, _refs(), k).graph)
