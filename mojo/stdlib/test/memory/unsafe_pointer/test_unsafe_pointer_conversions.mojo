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

from memory.unsafe_pointer import (
    UnsafePointer as UnsafePointerV1,
    UnsafePointerV2,
)

from testing import TestSuite


# V2 -> V1 tests


fn v1_mutable(_p: UnsafePointerV1[Int, mut=True, origin=_]):
    pass


fn v1_immutable(_p: UnsafePointerV1[Int, mut=False, origin=_]):
    pass


fn v1_mutable_any(_p: UnsafePointerV1[Int, mut=True, origin=MutAnyOrigin]):
    pass


fn v1_immutable_any(_p: UnsafePointerV1[Int, mut=False, origin=ImmutAnyOrigin]):
    pass


fn v1_unbound(_p: UnsafePointerV1[Int, **_]):
    pass


def test_v2_mutable_converts_to_v1():
    var x = 42
    var p = UnsafePointerV2(to=x)

    v1_mutable(p)
    v1_immutable(p)
    v1_mutable_any(p)
    v1_immutable_any(p)
    v1_unbound(p)


def test_v2_immutable_converts_to_v1():
    var x = 42
    var p = UnsafePointerV2(to=x).as_immutable()

    v1_mutable(p)
    v1_immutable(p)
    v1_mutable_any(p)
    v1_immutable_any(p)
    v1_unbound(p)


def test_v2_mutable_any_converts_to_v1():
    var x = 42
    var p = UnsafePointerV2(to=x).as_any_origin()

    v1_mutable(p)
    v1_immutable(p)
    v1_mutable_any(p)
    v1_immutable_any(p)
    v1_unbound(p)


def test_v2_immutable_any_converts_to_v1():
    var x = 42
    var p = UnsafePointerV2(to=x).as_immutable().as_any_origin()

    v1_mutable(p)
    v1_immutable(p)
    v1_mutable_any(p)
    v1_immutable_any(p)
    v1_unbound(p)


# V1 -> V2 tests


fn v2_mutable(_p: UnsafeMutPointer[Int]):
    pass


fn v2_immutable(_p: UnsafeImmutPointer[Int]):
    pass


fn v2_mutable_any(_p: UnsafePointerV2[Int, MutAnyOrigin]):
    pass


fn v2_immutable_any(_p: UnsafePointerV2[Int, ImmutAnyOrigin]):
    pass


fn v2_unbound(_p: UnsafePointerV2[Int, **_]):
    pass


def test_v1_mutable_converts_to_v2():
    var x = 42
    var p = UnsafePointerV1(to=x)

    v2_mutable(p)
    v2_immutable(p)
    v2_mutable_any(p)
    v2_immutable_any(p)
    v2_unbound(p)


def test_v1_immutable_converts_to_v2():
    var x = 42
    var p = UnsafePointerV1(to=x).as_immutable()

    v2_mutable(p)
    v2_immutable(p)
    v2_mutable_any(p)
    v2_immutable_any(p)
    v2_unbound(p)


def test_v1_mutable_any_converts_to_v2():
    var x = 42
    var p = UnsafePointerV1(to=x).as_any_origin()

    v2_mutable(p)
    v2_immutable(p)
    v2_mutable_any(p)
    v2_immutable_any(p)
    v2_unbound(p)


def test_v1_immutable_any_converts_to_v2():
    var x = 42
    var p = UnsafePointerV1(to=x).as_immutable().as_any_origin()

    v2_mutable(p)
    v2_immutable(p)
    v2_mutable_any(p)
    v2_immutable_any(p)
    v2_unbound(p)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
