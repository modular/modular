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

"""Shared helpers for benchmark entrypoints."""

from __future__ import annotations

import resource

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def get_tokenizer(
    pretrained_model_name_or_path: str,
    model_max_length: int | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a tokenizer for a benchmark model."""
    tokenizer_kwargs: dict[str, bool | int] = {
        "trust_remote_code": trust_remote_code,
    }
    if model_max_length is not None:
        tokenizer_kwargs["model_max_length"] = model_max_length
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        **tokenizer_kwargs,
    )


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit: int = 65535) -> None:
    """Raise the open-file soft limit when benchmarks need many sockets."""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as error:
            print(f"Fail to set RLIMIT_NOFILE: {error}")


def print_section(title: str, char: str = "-") -> None:
    """Print a formatted section header for benchmark output."""
    print("{s:{c}^{n}}".format(s=title, n=50, c=char))
