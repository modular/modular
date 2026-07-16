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
"""Tests for benchmark_shared.serving_result_output."""

from __future__ import annotations

from max.benchmark.benchmark_shared.serving_result_output import (
    elide_data_uris_in_string,
)


def test_elide_data_uris_in_string() -> None:
    """Test that elide_data_uris_in_string correctly elides base64 data URIs."""

    # fmt: off

    # Basic case
    sample = "'image': 'data:image/jpeg;base64,/9j/4AAQSASDEEAE'"
    expected = "'image': 'data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)...'"
    assert elide_data_uris_in_string(sample) == expected

    # Two data URIs in a single string
    sample = "data:image/jpeg;base64,/9j/4AAQSASDEEAE + data:image/jpeg;base64,/9j/4AAQSASDEEAE"
    expected = "data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)... + data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)..."
    assert elide_data_uris_in_string(sample) == expected

    # Still elides even if it results in longer string
    sample = "data:image/jpeg;base64,ABC"
    expected = "data:image/jpeg;base64,...(hash: b5d4045c, 3 bytes)..."
    assert elide_data_uris_in_string(sample) == expected

    # Does not elide if invalid characters in data
    sample = "data:image/jpeg;base64,ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧"
    expected = "data:image/jpeg;base64,ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧"
    assert elide_data_uris_in_string(sample) == expected

    # Does not elide if data uri type is empty
    sample = "data:;base64,ABC"
    expected = "data:;base64,ABC"
    assert elide_data_uris_in_string(sample) == expected

    # `data:` is present in string but not part of data uri
    sample = "Here is some data: 'data:image/jpeg;base64,AAAAAAAAASTUFF=='"
    expected = "Here is some data: 'data:image/jpeg;base64,...(hash: 6c6e1584, 16 bytes)...'"
    assert elide_data_uris_in_string(sample) == expected

    # `;base64` is present in string but not part of data uri
    sample = ";base64"
    expected = ";base64"
    assert elide_data_uris_in_string(sample) == expected

    # String is empty
    sample = ""
    expected = ""
    assert elide_data_uris_in_string(sample) == expected

    # fmt: on
