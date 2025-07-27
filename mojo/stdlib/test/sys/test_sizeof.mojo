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

from sys import sizeof

from testing import assert_equal


def main():
    assert_equal(sizeof[DType.int8](), 1)
    assert_equal(sizeof[DType.int16](), 2)
    assert_equal(sizeof[DType.int32](), 4)
    assert_equal(sizeof[DType.int64](), 8)
    assert_equal(sizeof[DType.float32](), 4)
    assert_equal(sizeof[DType.float64](), 8)
    assert_equal(sizeof[DType.bool](), 1)
    assert_equal(sizeof[DType.index](), 8)
    assert_equal(sizeof[DType.float8_e4m3fn](), 1)
    assert_equal(sizeof[DType.float8_e5m2fnuz](), 1)
    assert_equal(sizeof[DType.float8_e4m3fnuz](), 1)
    assert_equal(sizeof[DType.bfloat16](), 2)
    assert_equal(sizeof[DType.float16](), 2)
    assert_equal(sizeof[DType.invalid](), 0)
