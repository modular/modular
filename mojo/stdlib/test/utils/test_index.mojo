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

from testing import assert_equal

from utils import Index, IndexList


def test_basics():
    assert_equal(IndexList[2](1, 2), IndexList[2](1, 2))
    assert_equal(IndexList[3](1, 2, 3), IndexList[3](1, 2, 3))
    assert_equal(String(IndexList[3](1, 2, 3)), "(1, 2, 3)")
    assert_equal(IndexList[3](1, 2, 3)[2], 3)


def test_cast():
    assert_equal(
        String(IndexList[1](1)),
        "(1,)",
    )
    assert_equal(
        String(IndexList[2](1, 2).cast[DType.int32]()),
        "(1, 2)",
    )
    assert_equal(
        String(IndexList[2, element_type = DType.int32](1, 2)),
        "(1, 2)",
    )
    assert_equal(
        String(IndexList[2, element_type = DType.int64](1, 2)),
        "(1, 2)",
    )
    assert_equal(
        String(
            IndexList[2, element_type = DType.int32](1, -2).cast[DType.int64]()
        ),
        "(1, -2)",
    )
    assert_equal(
        String(IndexList[2, element_type = DType.int32](1, 2)),
        "(1, 2)",
    )
    alias s = String(
        IndexList[2, element_type = DType.int32](1, 2).cast[DType.int64]()
    )
    assert_equal(s, "(1, 2)")


def test_index():
    assert_equal(String(Index[dtype = DType.int64](1, 2, 3)), "(1, 2, 3)")
    assert_equal(String(Index[dtype = DType.int32](1, 2, 3)), "(1, 2, 3)")
    assert_equal(String(Index[dtype = DType.uint32](1, 2, 3)), "(1, 2, 3)")


def main():
    test_basics()
    test_cast()
    test_index()
