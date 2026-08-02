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
"""Host-side tests for `Span`'s `address_space` parameter.

Non-generic address spaces have no defined semantics in host code (LLVM
assigns no meaning to them on CPU targets), so the shared-memory spans here
are empty and never dereferenced: these tests only exercise the type-level
machinery — construction, address-space preservation through slicing and
conversions, and overload dispatch. Real shared-memory reads and writes are
covered by the GPU launch test in `max/mojo/test/gpu/collections/`.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

comptime SharedTile = Span[
    mut=True, Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


# A kernel-style helper restricted to shared-memory tiles.
def _tile_sum[
    dtype: DType, //
](tile: Span[Scalar[dtype], _, address_space=AddressSpace.SHARED]) -> Scalar[
    dtype
]:
    var acc = Scalar[dtype](0)
    for i in range(len(tile)):
        acc += tile[i]
    return acc


# Address-space-generic: one body, reusable for global, shared and local tiles.
def _scale_in_place[
    dtype: DType, //
](
    buf: Span[mut=True, Scalar[dtype], _, address_space=_],
    factor: Scalar[dtype],
):
    for i in range(len(buf)):
        buf[i] *= factor


def test_shared_span_construction() raises:
    var tile = SharedTile()
    assert_equal(len(tile), 0)
    assert_false(Bool(tile))
    assert_equal(type_of(tile).address_space, AddressSpace.SHARED)
    assert_equal(type_of(tile.unsafe_ptr()).address_space, AddressSpace.SHARED)


def test_shared_span_helper_dispatch() raises:
    var tile = SharedTile()
    assert_equal(_tile_sum(tile), 0.0)
    _scale_in_place(tile, 2.0)
    assert_false(2.0 in tile)


def test_shared_span_slicing_preserves_address_space() raises:
    comptime Sliced = type_of(SharedTile()[0:0])
    assert_equal(Sliced.address_space, AddressSpace.SHARED)
    comptime SubSpan = type_of(SharedTile().unsafe_subspan(offset=0, length=0))
    assert_equal(SubSpan.address_space, AddressSpace.SHARED)


def test_as_imm_preserves_address_space() raises:
    var tile = SharedTile()
    var ro = tile.as_imm()
    assert_equal(type_of(ro).address_space, AddressSpace.SHARED)
    assert_equal(_tile_sum(ro), 0.0)


def test_generic_address_space_helper_on_host_memory() raises:
    var host: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var hspan = Span(host)
    _scale_in_place(hspan, 3.0)
    assert_equal(hspan[0], 3.0)
    assert_equal(hspan[3], 12.0)


def test_generic_spans_keep_full_surface() raises:
    var host: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    assert_equal(type_of(Span(host)).address_space, AddressSpace.GENERIC)
    assert_true(2.0 in Span(host))

    var strs: List[String] = ["a", "b"]
    assert_equal(String(Span(strs)), "[a, b]")

    var copy: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    Span(copy).copy_from(Span(host))
    assert_equal(copy[2], 3.0)

    Span(copy).fill(7.0)
    assert_equal(copy[0], 7.0)
    assert_true(Span(copy) == Span(copy))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
