# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Helper functions for asserting in ops."""

from collections.abc import Iterable

from ..value import BufferValue, TensorValue


def indent(
    lines: Iterable[str], level: int = 1, indent: str = "    "
) -> Iterable[str]:
    for line in lines:
        yield (indent * level) + line


def assert_same_device(
    *values: TensorValue | BufferValue,
    **named_values: TensorValue | BufferValue,
) -> None:
    named_values = {
        **{str(i): value for i, value in enumerate(values)},
        **named_values,
    }
    if len({v.device for v in named_values.values()}) > 1:
        raise ValueError(
            "Input values must be on the same device\n"
            + "\n".join(
                indent(
                    f"{name}: {value.type}"
                    for name, value in named_values.items()
                )
            )
        )
