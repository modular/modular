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
"""Pins the Hy4 target adapter's registration and graph inputs."""

from __future__ import annotations

from max.pipelines.architectures.hy_v4.spec_adapters import (
    ADAPTER,
    HYV4MTPProposer,
    HYV4Target,
)
from max.pipelines.speculative import spec_adapters


def test_is_the_registered_adapter() -> None:
    assert spec_adapters.primary_kv("HYV4ForCausalLM", "mtp") == "mla"


def test_adds_no_unnamed_inputs() -> None:
    """The host length rides the named ``batch_context_lengths``, which the
    target and proposer opt into, so an exported fused graph names every
    input."""
    assert list(ADAPTER.trailing_input_types(None)) == []
    assert HYV4Target.takes_distributed_inputs
    assert HYV4MTPProposer.takes_distributed_inputs
