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

import pytest
from max.benchmark.benchmark_shared.utils import (
    get_tokenizer,
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
