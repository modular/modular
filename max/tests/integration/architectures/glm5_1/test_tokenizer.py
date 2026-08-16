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

"""Tests for GLM's reasoning-effort normalization.

These mirror the one line of GLM's chat template that consumes the value::

    {%- set effective_reasoning_effort =
        'high' if reasoning_effort is defined and reasoning_effort == 'high'
        else 'max' -%}

so ``"high"`` is the only string that selects the lower of the template's two
thinking rungs and everything else falls through to the higher one. The
normalization therefore sends every effort except ``"none"`` (thinking off),
``"max"`` (the native top rung) and ``"xhigh"`` (OpenRouter's name for it) to
``"high"``.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures.glm5_1.tokenizer import (
    normalize_glm_reasoning_effort,
)


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_openai_ladder_selects_the_lower_rung(effort: str) -> None:
    """The whole OpenAI ladder maps to High; only "max" reaches Max."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": effort}) == {
        "reasoning_effort": "high"
    }


def test_max_outranks_the_ladder() -> None:
    """The template's top rung stays above every OpenAI effort."""
    top = normalize_glm_reasoning_effort({"reasoning_effort": "max"})
    high = normalize_glm_reasoning_effort({"reasoning_effort": "high"})
    assert top["reasoning_effort"] == "max"
    assert high["reasoning_effort"] == "high"


@pytest.mark.parametrize("alias", ["xhigh", "XHigh", "  xhigh  "])
def test_openrouter_xhigh_reaches_the_top_rung(alias: str) -> None:
    """OpenRouter advertises ``["xhigh", "high"]`` for this model, so ``xhigh``
    is the only way one of its clients can ask for GLM's upper rung. Falling
    through to the ladder would hand the highest request the lower rung."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": alias}) == {
        "reasoning_effort": "max"
    }


def test_xhigh_outranks_high() -> None:
    """The pair must stay ordered the way OpenRouter documents them."""
    xhigh = normalize_glm_reasoning_effort({"reasoning_effort": "xhigh"})
    high = normalize_glm_reasoning_effort({"reasoning_effort": "high"})
    assert xhigh["reasoning_effort"] == "max"
    assert high["reasoning_effort"] == "high"
    assert xhigh["reasoning_effort"] != high["reasoning_effort"]


def test_unrecognized_effort_degrades_to_the_lower_rung() -> None:
    """An unknown value must not silently max out reasoning (the template's
    own fallthrough behavior for anything that is not exactly "high")."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": "extreme"}) == {
        "reasoning_effort": "high"
    }


def test_none_disables_thinking() -> None:
    """The template drops the effort line entirely once the toggle is off."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": "none"}) == {
        "reasoning_effort": "none",
        "enable_thinking": False,
        "thinking": False,
    }


@pytest.mark.parametrize("toggle", ["enable_thinking", "thinking"])
def test_none_does_not_override_an_explicit_toggle(toggle: str) -> None:
    """A caller that set the toggle itself wins over the effort."""
    result = normalize_glm_reasoning_effort(
        {"reasoning_effort": "none", toggle: True}
    )
    assert result[toggle] is True
    assert "enable_thinking" not in result or result["enable_thinking"] is True


def test_absent_effort_is_left_alone() -> None:
    """No effort means the template's own default applies, untouched."""
    assert normalize_glm_reasoning_effort({"enable_thinking": True}) == {
        "enable_thinking": True
    }
    assert normalize_glm_reasoning_effort({}) == {}


def test_max_targets_the_template_natively() -> None:
    """``"max"`` is the template's own top rung and passes through unchanged."""
    assert normalize_glm_reasoning_effort({"reasoning_effort": "max"}) == {
        "reasoning_effort": "max"
    }


@pytest.mark.parametrize("effort", ["MAX", "  max  ", "Max"])
def test_effort_is_matched_case_and_space_insensitively(effort: str) -> None:
    assert normalize_glm_reasoning_effort({"reasoning_effort": effort}) == {
        "reasoning_effort": "max"
    }


def test_non_string_effort_is_ignored() -> None:
    assert normalize_glm_reasoning_effort({"reasoning_effort": None}) == {
        "reasoning_effort": None
    }


def test_other_options_are_preserved() -> None:
    result = normalize_glm_reasoning_effort(
        {"reasoning_effort": "low", "add_generation_prompt": True}
    )
    assert result["add_generation_prompt"] is True


def test_input_is_not_mutated() -> None:
    options = {"reasoning_effort": "low"}
    normalize_glm_reasoning_effort(options)
    assert options == {"reasoning_effort": "low"}
