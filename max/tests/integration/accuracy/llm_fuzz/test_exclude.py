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

import pytest
from helpers import normalize_test_id, parse_exclude

_SCENARIOS = {"constrained_decoding", "tc_basics", "zombie_horde"}


def test_whole_scenarios_and_test_ids_are_separated() -> None:
    groups, tests = parse_exclude(
        "zombie_horde,tc_basics:parallel_tc_two_tools", _SCENARIOS
    )

    assert groups == {"zombie_horde"}
    assert tests == {"tc_basics:parallel_tc_two_tools"}


def test_comma_inside_a_test_name_is_not_a_separator() -> None:
    # The whitespace-stripped form the .sh loader produces, so the fragment
    # after the comma still carries the ``::variant`` suffix (and a colon).
    spec = (
        "constrained_decoding:anyOf[7]:nestedanyOf,"
        "tocheckvalidationsemantics::tool_auto_strict,"
        "tc_basics:parallel_tc_two_tools"
    )

    groups, tests = parse_exclude(spec, _SCENARIOS)

    assert groups == set()
    assert tests == {
        "constrained_decoding:anyOf[7]:nestedanyOf,"
        "tocheckvalidationsemantics::tool_auto_strict",
        "tc_basics:parallel_tc_two_tools",
    }


def test_entry_with_spaces_matches_whitespace_free_runtime_id() -> None:
    # The Mammoth loader keeps interior spaces; the runtime key is
    # whitespace-free. Both must normalize to the same id.
    _, tests = parse_exclude(
        "constrained_decoding:anyOf[7]:nested anyOf, to check semantics"
        "::tool_auto_strict",
        _SCENARIOS,
    )

    runtime_id = normalize_test_id(
        "constrained_decoding:anyOf[7]:nested anyOf, to check semantics"
        "::tool_auto_strict"
    )
    assert runtime_id in tests


def test_without_known_scenarios_every_comma_splits() -> None:
    _, tests = parse_exclude("a:b,c:d")

    assert tests == {"a:b", "c:d"}


def test_unknown_scenario_fragment_is_glued_but_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Indistinguishable from a comma inside a test name, so it is glued
    # but must warn.
    _, tests = parse_exclude(
        "tc_basics:parallel_tc_two_tools,stale_scenario:x", _SCENARIOS
    )

    assert tests == {"tc_basics:parallel_tc_two_tools,stale_scenario:x"}
    assert "stale_scenario:x" in capsys.readouterr().err


def test_unknown_leading_entry_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, tests = parse_exclude(
        "typo_scenario:x,tc_basics:parallel_tc_two_tools", _SCENARIOS
    )

    assert tests == {"typo_scenario:x", "tc_basics:parallel_tc_two_tools"}
    assert "typo_scenario:x" in capsys.readouterr().err
