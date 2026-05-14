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
from unittest.mock import MagicMock

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
    wait_for_server_ready,
)
from pytest_mock import MockerFixture


def test_get_tokenizer_passes_model_max_length_when_provided(
    mocker: MockerFixture,
) -> None:
    from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
    )
    mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=MagicMock(architectures=[]),
    )

    tokenizer = get_tokenizer(
        "repo/model",
        revision="abc123",
        model_max_length=4096,
        trust_remote_code=True,
    )

    assert tokenizer is from_pretrained.return_value
    assert tokenizer._resolved_revision == "abc123"
    from_pretrained.assert_called_once_with(
        "repo/model",
        model_max_length=4096,
        trust_remote_code=True,
        revision="abc123",
    )


def test_get_tokenizer_omits_model_max_length_when_unspecified(
    mocker: MockerFixture,
) -> None:
    from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
    )
    mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=MagicMock(architectures=[]),
    )

    tokenizer = get_tokenizer(
        "repo/model", revision=None, trust_remote_code=False
    )

    assert tokenizer is from_pretrained.return_value
    assert tokenizer._resolved_revision is None
    from_pretrained.assert_called_once_with(
        "repo/model",
        trust_remote_code=False,
        revision=None,
    )


def test_set_ulimit_updates_soft_limit_when_needed(
    mocker: MockerFixture,
) -> None:
    getrlimit = mocker.patch("resource.getrlimit", return_value=(1024, 65535))
    setrlimit = mocker.patch("resource.setrlimit")

    set_ulimit(target_soft_limit=4096)

    getrlimit.assert_called_once_with(resource.RLIMIT_NOFILE)
    setrlimit.assert_called_once_with(
        resource.RLIMIT_NOFILE,
        (4096, 65535),
    )


def test_set_ulimit_skips_update_when_soft_limit_is_high_enough(
    mocker: MockerFixture,
) -> None:
    mocker.patch("resource.getrlimit", return_value=(8192, 65535))
    setrlimit = mocker.patch("resource.setrlimit")

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


# ---- wait_for_server_ready ----


def _mock_response(status: int) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.status = status
    response.__exit__.return_value = False
    return response


def test_wait_for_server_ready_returns_on_200(mocker: MockerFixture) -> None:
    mocker.patch("urllib.request.urlopen", return_value=_mock_response(200))
    mocker.patch("time.monotonic", side_effect=[100.0, 101.5])
    sleep = mocker.patch("time.sleep")

    elapsed = wait_for_server_ready("localhost", 8000, timeout_s=60)

    assert elapsed == pytest.approx(1.5)
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "first_attempt",
    [503, 404, 204, ConnectionRefusedError()],
)
def test_wait_for_server_ready_polls_past_non_ready(
    mocker: MockerFixture,
    first_attempt: int | Exception,
) -> None:
    urlopen = mocker.patch("urllib.request.urlopen")
    if isinstance(first_attempt, Exception):
        urlopen.side_effect = [first_attempt, _mock_response(200)]
    else:
        urlopen.side_effect = [
            _mock_response(first_attempt),
            _mock_response(200),
        ]
    mocker.patch("time.monotonic", side_effect=[100.0, 101.0, 105.0])
    sleep = mocker.patch("time.sleep")

    elapsed = wait_for_server_ready("localhost", 8000, timeout_s=60)

    assert elapsed == pytest.approx(5.0)
    assert urlopen.call_count == 2
    sleep.assert_called_once_with(5.0)


def test_wait_for_server_ready_raises_on_timeout(
    mocker: MockerFixture,
) -> None:
    mocker.patch("urllib.request.urlopen", return_value=_mock_response(503))
    mocker.patch("time.monotonic", side_effect=[100.0, 200.0])
    sleep = mocker.patch("time.sleep")

    with pytest.raises(RuntimeError, match="not ready"):
        wait_for_server_ready("localhost", 8000, timeout_s=5)

    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("first_attempt", "expects_raise"),
    [
        (200, False),
        (503, True),
        (ConnectionRefusedError(), True),
    ],
)
def test_wait_for_server_ready_zero_timeout_tries_once(
    mocker: MockerFixture,
    first_attempt: int | Exception,
    expects_raise: bool,
) -> None:
    urlopen = mocker.patch("urllib.request.urlopen")
    if isinstance(first_attempt, Exception):
        urlopen.side_effect = [first_attempt]
    else:
        urlopen.return_value = _mock_response(first_attempt)
    mocker.patch("time.monotonic", side_effect=[100.0, 100.0])
    sleep = mocker.patch("time.sleep")

    if expects_raise:
        with pytest.raises(RuntimeError):
            wait_for_server_ready("localhost", 8000, timeout_s=0)
    else:
        wait_for_server_ready("localhost", 8000, timeout_s=0)

    assert urlopen.call_count == 1
    sleep.assert_not_called()
