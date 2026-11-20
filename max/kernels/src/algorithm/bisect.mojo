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
"""Binary search: `bisect_left` / `bisect_right` (and `bisect`, their
right-biased alias) mirror Python's `bisect` module.

Three call shapes, all over the index range `[0, hi)`, where `hi` is a
runtime `Int`:

- `bisect_left(values, x)` on an `ImmSpan` of non-decreasing values.
- `bisect_left(hi, x, load)` where `load(i)` produces `values[i]`, for values
  behind a device load or a computed address rather than a span.
- `bisect_left(hi, probe)` where `probe(i)` reports whether index `i` sorts
  before the split, for values ordered in some other way (descending, ordered
  by a derived key, or not stored at all). This is the partition-point form
  that the other two are specializations of.

The `lo` end of the range is always 0; callers searching a suffix fold the
offset into `load` / `probe` and add it back to the result.
"""

from std.collections import ImmSpan


# ===-----------------------------------------------------------------------===#
# The shared loop
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _bisect[
    left: Bool, T: Comparable & Copyable & Deinitable, F: def(Int) -> T
](hi: Int, x: T, load: F) -> Int:
    """Returns the first index in `[0, hi)` whose value is not before `x`.

    Both variants share this loop: `left` picks `< x` (below the split) over
    `<= x` (at or below it) as the predicate that advances `lo`.

    Parameters:
        left: Search below the split (`< x`) rather than at or below it.
        T: The value type; ordered, and cheap to compare.
        F: The load closure's type.

    Args:
        hi: One past the last index to search. Every index in `[0, hi)` is
            probed, so `hi` must not exceed the value range.
        x: The key to locate.
        load: Produces the value at an index.

    Returns:
        The first index whose value is not before `x`, which is `hi` when no
        value is at or past `x`.
    """
    var lo = 0
    var end = hi
    while lo < end:
        var mid = (lo + end) >> 1
        comptime if left:
            if load(mid) < x:
                lo = mid + 1
            else:
                end = mid
        else:
            if load(mid) <= x:
                lo = mid + 1
            else:
                end = mid
    return lo


# ===-----------------------------------------------------------------------===#
# bisect_left
# ===-----------------------------------------------------------------------===#


@inline(.always)
def bisect_left[
    T: Comparable & Copyable & Deinitable, F: def(Int) -> T
](hi: Int, x: T, load: F) -> Int:
    """Returns the index at which to insert `x` in a sorted range of `hi`
    values, before any equal values.

    Parameters:
        T: The value type; ordered, and cheap to compare.
        F: The load closure's type.

    Args:
        hi: One past the last index to search. Every index in `[0, hi)` is
            probed, so `hi` must not exceed the value range.
        x: The key to locate.
        load: Produces the value at an index.

    Returns:
        The first index whose value is greater than or equal to `x`, which is
        `hi` when every value is below `x`.
    """
    return _bisect[True](hi, x, load)


@inline(.always)
def bisect_left[
    T: Comparable & Copyable & Deinitable
](values: ImmSpan[T, _], x: T) -> Int:
    """Returns the index at which to insert `x` in `values`, before any equal
    values.

    Args:
        values: A non-decreasing span of values.
        x: The key to locate.

    Returns:
        The first index whose value is greater than or equal to `x`, which is
        `len(values)` when every value is below `x`.
    """

    return _bisect[True](len(values), x, lambda (i: Int) -> T: values[i].copy())


@inline(.always)
def bisect_left[F: def(Int) -> Bool](hi: Int, probe: F) -> Int:
    """Returns the first index whose probe reports it sorts at or past the
    split.

    The partition-point form of `bisect_left`, for a range whose order is
    expressed by the probe itself rather than by the values: pass
    `probe(i) = values[i] < x` for an ascending range and
    `probe(i) = values[i] > x` for a descending one.

    Parameters:
        F: The probe closure's type.

    Args:
        hi: One past the last index to search. Every index in `[0, hi)` is
            probed, so `hi` must not exceed the range.
        probe: Reports whether index `i` sorts before the split. Must be True
            on `[0, k)` and False on `[k, hi)` for the returned `k`.

    Returns:
        The first index whose probe returns False, which is `hi` when the
        probe is True throughout.
    """
    var lo = 0
    var end = hi
    while lo < end:
        var mid = (lo + end) >> 1
        if probe(mid):
            lo = mid + 1
        else:
            end = mid
    return lo


# ===-----------------------------------------------------------------------===#
# bisect_right / bisect
# ===-----------------------------------------------------------------------===#


@inline(.always)
def bisect_right[
    T: Comparable & Copyable & Deinitable, F: def(Int) -> T
](hi: Int, x: T, load: F) -> Int:
    """Returns the index at which to insert `x` in a sorted range of `hi`
    values, after any equal values.

    Parameters:
        T: The value type; ordered, and cheap to compare.
        F: The load closure's type.

    Args:
        hi: One past the last index to search. Every index in `[0, hi)` is
            probed, so `hi` must not exceed the value range.
        x: The key to locate.
        load: Produces the value at an index.

    Returns:
        The first index whose value is greater than `x`, which is `hi` when no
        value exceeds `x`.
    """
    return _bisect[False](hi, x, load)


@inline(.always)
def bisect_right[
    T: Comparable & Copyable & Deinitable
](values: ImmSpan[T, _], x: T) -> Int:
    """Returns the index at which to insert `x` in `values`, after any equal
    values.

    Args:
        values: A non-decreasing span of values.
        x: The key to locate.

    Returns:
        The first index whose value is greater than `x`, which is
        `len(values)` when no value exceeds `x`.
    """

    return _bisect[False](
        len(values), x, lambda (i: Int) -> T: values[i].copy()
    )


@inline(.always)
def bisect[
    T: Comparable & Copyable & Deinitable, F: def(Int) -> T
](hi: Int, x: T, load: F) -> Int:
    """Right-biased `bisect_left`: the insertion point after equal values.

    Parameters:
        T: The value type; ordered, and cheap to compare.
        F: The load closure's type.

    Args:
        hi: One past the last index to search. Every index in `[0, hi)` is
            probed, so `hi` must not exceed the value range.
        x: The key to locate.
        load: Produces the value at an index.

    Returns:
        The first index whose value is greater than `x`.
    """
    return bisect_right(hi, x, load)


@inline(.always)
def bisect[
    T: Comparable & Copyable & Deinitable
](values: ImmSpan[T, _], x: T) -> Int:
    """Right-biased `bisect_left`: the insertion point after equal values.

    Args:
        values: A non-decreasing span of values.
        x: The key to locate.

    Returns:
        The first index whose value is greater than `x`.
    """
    return bisect_right(values, x)
