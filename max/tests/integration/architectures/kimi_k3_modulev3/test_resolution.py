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
"""What a Kimi K3 checkpoint resolves to now that ModuleV3 is the only one.

The published checkpoint names ``KimiK3ForConditionalGeneration`` in its
``architectures[0]``, and the graph-API architecture used to answer for that
name. ModuleV3 is registered under the ``_ModuleV3`` suffix, which a plain
lookup reaches only through its fallback to the single registered variant --
so the fallback is what keeps every existing K3 checkpoint loadable, and it
is pinned here rather than left to a serving run to discover.

The speculative pairing has a second gate on top of the name lookup: the
graph-API K3 refused the Inferact DSpark drafter, and the ModuleV3 K3 is the
one that serves it. With the graph-API registration gone the lookup would
reach ModuleV3 either way, but the pairing's own routing (which keys on the
drafter's name) must still land there -- so this pins the pairing too.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.kimi_k3_modulev3.arch import (
    kimi_k3_modulev3_arch,
)
from max.pipelines.lib.arch_lookup import ARCH_LOOKUP, ArchLookup
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _builds_module,
    _target_architecture,
)

_CHECKPOINT_ARCH = "KimiK3ForConditionalGeneration"
"""What the checkpoint's ``config.json`` declares, unchanged by this port."""

_MODULEV3_ARCH = "KimiK3ForConditionalGeneration_ModuleV3"

_DSPARK_DRAFT_ARCH = "K3DSparkModel"
"""What the Inferact draft checkpoint's ``architectures[0]`` declares."""


@pytest.fixture(autouse=True)
def _registered_architectures() -> None:
    """Populates the global lookup, which importing ``max.pipelines`` does.

    Importing ``max.pipelines.lib.arch_lookup`` above imports the
    ``max.pipelines`` package, whose ``__init__`` already calls
    ``register_all_models()``, so this is an idempotent guarantee rather than
    a requirement: it keeps the table populated however this module's imports
    are later rearranged.
    """
    register_all_models()


def test_the_graph_api_variant_is_absent_from_this_table() -> None:
    """The premise: this target depends on the ModuleV3 package alone.

    So the table these tests read is the one a deployment has once the
    graph-API K3 is deleted, not one that happens to answer the same way.
    """
    assert ARCH_LOOKUP.resolve(_CHECKPOINT_ARCH) is None
    assert ARCH_LOOKUP.resolve(_MODULEV3_ARCH) is not None


def test_modulev3_is_registered_under_the_suffixed_name() -> None:
    """Guards the fallback below against passing on nothing."""
    registered = ARCH_LOOKUP.find(_MODULEV3_ARCH)

    assert registered is not None
    assert registered.name == _MODULEV3_ARCH


def test_the_checkpoint_name_resolves_without_the_variant_suffix() -> None:
    """With only ModuleV3 registered, its name resolves with no preference.

    A fresh lookup holds only the ModuleV3 registration, which is the table a
    deployment has once ``max_private/kimi_k3`` is not importable. The plain
    ``find`` is what config resolution calls for a checkpoint, and it must
    answer with the one variant there is rather than ``None``.
    """
    lookup = ArchLookup()
    lookup.register(kimi_k3_modulev3_arch)

    assert lookup.find(_MODULEV3_ARCH) is kimi_k3_modulev3_arch
    assert lookup.find(_CHECKPOINT_ARCH) is kimi_k3_modulev3_arch
    assert lookup.find(_CHECKPOINT_ARCH, prefer_module_v3=True) is (
        kimi_k3_modulev3_arch
    )


def test_the_unknown_name_is_still_unresolved() -> None:
    """The fallback is by variant, not a catch-all: a name with no variant
    registered stays unresolved."""
    lookup = ArchLookup()
    lookup.register(kimi_k3_modulev3_arch)

    assert lookup.find("KimiK2ForConditionalGeneration") is None


def test_the_modulev3_target_builds_its_module() -> None:
    """What makes the speculative pairing accept it as a target at all."""
    assert _builds_module(kimi_k3_modulev3_arch)


def test_the_spec_lookup_picks_modulev3_for_the_inferact_drafter() -> None:
    """The pairing's target lookup, keyed on the drafter's architecture name.

    ``prefer_module_v3`` is false here, as config resolution passes it for a
    checkpoint that names the plain architecture: the drafter's name is what
    has to move the answer onto the ModuleV3 target.
    """
    target = _target_architecture(
        _CHECKPOINT_ARCH, draft_arch=_DSPARK_DRAFT_ARCH
    )

    assert target is not None
    assert target.name == _MODULEV3_ARCH
    assert target is ARCH_LOOKUP.find(_MODULEV3_ARCH)


def test_the_spec_lookup_still_finds_the_target_with_the_preference() -> None:
    """A caller that already prefers ModuleV3 lands on the same target."""
    target = _target_architecture(
        _CHECKPOINT_ARCH,
        prefer_module_v3=True,
        draft_arch=_DSPARK_DRAFT_ARCH,
    )

    assert target is not None
    assert target.name == _MODULEV3_ARCH
