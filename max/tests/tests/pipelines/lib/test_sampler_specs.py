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
"""The sampler graphs group only where the group will actually form.

Grouping skips the session's MEF store, so it has to be off wherever an
artifact would be reused or recorded, and the bitmask sampler's name follows
the group rather than the request for one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.lib.graph_specs import GraphRole, compile_roles_separately
from max.pipelines.lib.pipeline_variants.text_generation import sampler_specs
from max.pipelines.sampling.sampling_config import SamplingConfig


def _config(
    *,
    precompiled_mefs: Any = None,
    export_mefs: Any = None,
    needs_bitmask: bool = True,
) -> Any:
    """A stand-in carrying only the fields `sampler_specs` reads."""
    return SimpleNamespace(
        sampling=SamplingConfig(),
        needs_bitmask_constraints=needs_bitmask,
        runtime=SimpleNamespace(
            precompiled_mefs=precompiled_mefs, export_mefs=export_mefs
        ),
    )


def _specs(config: Any) -> dict[str, Any]:
    return sampler_specs(config, DeviceRef.CPU(), vocab_size=128)


def test_groups_when_no_artifact_is_reused_or_recorded() -> None:
    specs = _specs(_config())
    assert {spec.group for spec in specs.values()} == {"sampler"}
    names = {role: spec.name for role, spec in specs.items()}
    assert names[GraphRole.SAMPLER] != names[GraphRole.SAMPLER_BITMASK]


@pytest.mark.parametrize(
    "overrides",
    [
        {"precompiled_mefs": Path("/tmp/mefs")},
        {"export_mefs": Path("/tmp/mefs")},
    ],
    ids=["precompiled_mefs", "export_mefs"],
)
def test_a_mef_path_turns_grouping_off(overrides: dict[str, Any]) -> None:
    specs = _specs(_config(**overrides))
    assert {spec.group for spec in specs.values()} == {None}
    # The name a store recorded an artifact under must not move.
    assert {spec.name for spec in specs.values()} == {"top_k_sampler"}


def test_a_default_store_turns_grouping_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        InferenceSession, "default_mef_store", object(), raising=False
    )
    specs = _specs(_config())
    assert {spec.group for spec in specs.values()} == {None}
    assert {spec.name for spec in specs.values()} == {"top_k_sampler"}


def test_compiling_roles_separately_turns_grouping_off() -> None:
    """`max export` ignores groups, so naming for one would rename nothing."""
    with compile_roles_separately():
        specs = _specs(_config())
    assert {spec.group for spec in specs.values()} == {None}
    assert {spec.name for spec in specs.values()} == {"top_k_sampler"}


def test_one_variant_still_takes_the_plain_name() -> None:
    specs = _specs(_config(needs_bitmask=False))
    assert list(specs) == [GraphRole.SAMPLER]
    assert specs[GraphRole.SAMPLER].name == "top_k_sampler"
