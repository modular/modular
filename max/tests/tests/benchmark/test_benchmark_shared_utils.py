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

from __future__ import annotations

import resource
from unittest.mock import patch, sentinel

import numpy as np
import pytest
from max.benchmark.benchmark_shared.utils import (
    argmedian,
    get_tokenizer,
    int_or_none,
    is_castable_to_int,
    parse_comma_separated,
    print_section,
    set_ulimit,
)


def test_get_tokenizer_passes_model_max_length_when_provided() -> None:
    with patch(
        "max.benchmark.benchmark_shared.utils.AutoTokenizer.from_pretrained",
        return_value=sentinel.tokenizer,
    ) as from_pretrained:
        tokenizer = get_tokenizer(
            "repo/model",
            model_max_length=4096,
            trust_remote_code=True,
        )

    assert tokenizer is sentinel.tokenizer
    from_pretrained.assert_called_once_with(
        "repo/model",
        model_max_length=4096,
        trust_remote_code=True,
    )


def test_get_tokenizer_omits_model_max_length_when_unspecified() -> None:
    with patch(
        "max.benchmark.benchmark_shared.utils.AutoTokenizer.from_pretrained",
        return_value=sentinel.tokenizer,
    ) as from_pretrained:
        tokenizer = get_tokenizer("repo/model", trust_remote_code=False)

    assert tokenizer is sentinel.tokenizer
    from_pretrained.assert_called_once_with(
        "repo/model",
        trust_remote_code=False,
    )


def test_set_ulimit_updates_soft_limit_when_needed() -> None:
    with (
        patch(
            "max.benchmark.benchmark_shared.utils.resource.getrlimit",
            return_value=(1024, 65535),
        ) as getrlimit,
        patch(
            "max.benchmark.benchmark_shared.utils.resource.setrlimit"
        ) as setrlimit,
    ):
        set_ulimit(target_soft_limit=4096)

    getrlimit.assert_called_once_with(resource.RLIMIT_NOFILE)
    setrlimit.assert_called_once_with(
        resource.RLIMIT_NOFILE,
        (4096, 65535),
    )


def test_set_ulimit_skips_update_when_soft_limit_is_high_enough() -> None:
    with (
        patch(
            "max.benchmark.benchmark_shared.utils.resource.getrlimit",
            return_value=(8192, 65535),
        ),
        patch(
            "max.benchmark.benchmark_shared.utils.resource.setrlimit"
        ) as setrlimit,
    ):
        set_ulimit(target_soft_limit=4096)

    setrlimit.assert_not_called()


def test_print_section_formats_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_section(" Shared Utilities ", char="=")

    expected = "{s:{c}^{n}}\n".format(
        s=" Shared Utilities ",
        n=50,
        c="=",
    )
    assert capsys.readouterr().out == expected


# ---- parse_comma_separated ----


def test_parse_comma_separated_single_value() -> None:
    assert parse_comma_separated("42", int) == [42]


def test_parse_comma_separated_multiple_values() -> None:
    assert parse_comma_separated("1,2,4", int) == [1, 2, 4]


def test_parse_comma_separated_strips_whitespace() -> None:
    assert parse_comma_separated(" 10 , 20 , 30 ", float) == [10.0, 20.0, 30.0]


def test_parse_comma_separated_none_returns_default() -> None:
    assert parse_comma_separated(None, int) == [None]


def test_parse_comma_separated_none_with_explicit_default() -> None:
    assert parse_comma_separated(None, int, default=0) == [0]


def test_parse_comma_separated_with_int_or_none() -> None:
    assert parse_comma_separated("1,None,3", int_or_none) == [1, None, 3]


def test_parse_comma_separated_float_inf() -> None:
    result = parse_comma_separated("inf", float)
    assert result == [float("inf")]


def test_parse_comma_separated_mixed_floats() -> None:
    result = parse_comma_separated("10,inf,0.5", float)
    assert result == [10.0, float("inf"), 0.5]


# ---- is_castable_to_int ----


def test_is_castable_to_int_valid() -> None:
    assert is_castable_to_int("42") is True
    assert is_castable_to_int("-1") is True
    assert is_castable_to_int("0") is True


def test_is_castable_to_int_invalid() -> None:
    assert is_castable_to_int("hello") is False
    assert is_castable_to_int("3.14") is False
    assert is_castable_to_int("") is False


# ---- int_or_none ----


def test_int_or_none_integer() -> None:
    assert int_or_none("5") == 5
    assert int_or_none("-10") == -10


def test_int_or_none_none_literal() -> None:
    assert int_or_none("none") is None
    assert int_or_none("None") is None
    assert int_or_none("NONE") is None


def test_int_or_none_invalid_raises() -> None:
    with pytest.raises(ValueError):
        int_or_none("abc")


# ---- argmedian ----


def test_argmedian_odd_length() -> None:
    assert argmedian(np.array([10, 30, 20])) == 2


def test_argmedian_single_element() -> None:
    assert argmedian(np.array([7])) == 0


def test_argmedian_even_length_picks_nearest() -> None:
    idx = argmedian(np.array([1, 3, 5, 7]))
    assert idx in (1, 2)
