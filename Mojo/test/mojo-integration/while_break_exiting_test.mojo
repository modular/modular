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

# A `while True` that stores before it tests is exiting-test: the store also
# runs on the iteration that breaks, so the loop writes `n + 1` bytes. A raised
# `hlcf.for` tests its bound before the body and would drop that last store.

# RUN: %mojo %s | FileCheck %s
# `append_until` only miscompiled without debug info, so run both ways.
# RUN: %mojo-no-debug %s | FileCheck %s


# The test is `eq`, which is what a `while True` counting to a bound emits.
@no_inline
def fill_eq(p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var dest = p
    var count = n
    while True:
        dest[] = 7
        dest = dest.unsafe_offset(1)
        if count == 0:
            break
        count -= 1


# The same shape with the `gt` test the pass has always recognized.
@no_inline
def fill_gt(p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var dest = p
    var count = n
    while True:
        dest[] = 7
        dest = dest.unsafe_offset(1)
        if count > 0:
            pass
        else:
            break
        count -= 1


# Top-tested: the test precedes the store, so this one writes `n` bytes and
# does raise.
@no_inline
def fill_top_tested(p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var dest = p
    var count = n
    while True:
        if count == 0:
            break
        dest[] = 7
        dest = dest.unsafe_offset(1)
        count -= 1


# The decrement runs between the store and the test, so the test compares the
# updated value and the loop writes `n` bytes.
@no_inline
def fill_pre_increment(p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var dest = p
    var count = n
    while True:
        dest[] = 7
        dest = dest.unsafe_offset(1)
        count -= 1
        if count == 0:
            break


# Effects on both sides of the test: `a` takes the extra run, `b` does not.
@no_inline
def fill_both_sides(
    pa: Pointer[UInt8, MutUntrackedOrigin],
    pb: Pointer[UInt8, MutUntrackedOrigin],
    n: Int,
):
    var da = pa
    var db = pb
    var count = n
    while True:
        da[] = 7
        da = da.unsafe_offset(1)
        if count == 0:
            break
        db[] = 7
        db = db.unsafe_offset(1)
        count -= 1


# The append ahead of the test grows the list, so the copy of it at the end of
# the body has to store through the buffer its own reallocation returned.
@no_inline
def append_until(last: Int) -> List[Int]:
    var out = List[Int]()
    var k = 1
    while True:
        out.append(k)
        if k == last:
            break
        k += 1
    return out^


def written(buf: List[UInt8]) -> Int:
    var count = 0
    for b in buf:
        if b == 7:
            count += 1
    return count


def main():
    var eq = List[UInt8](length=16, fill=0)
    fill_eq(
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(eq.unsafe_ptr())
        ),
        4,
    )
    # CHECK: eq 5
    print("eq", written(eq))

    var gt = List[UInt8](length=16, fill=0)
    fill_gt(
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(gt.unsafe_ptr())
        ),
        4,
    )
    # CHECK: gt 5
    print("gt", written(gt))

    var top = List[UInt8](length=16, fill=0)
    fill_top_tested(
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(top.unsafe_ptr())
        ),
        4,
    )
    # CHECK: top 4
    print("top", written(top))

    var pre = List[UInt8](length=16, fill=0)
    fill_pre_increment(
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(pre.unsafe_ptr())
        ),
        4,
    )
    # CHECK: pre 4
    print("pre", written(pre))

    var a = List[UInt8](length=16, fill=0)
    var b = List[UInt8](length=16, fill=0)
    fill_both_sides(
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(a.unsafe_ptr())
        ),
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(b.unsafe_ptr())
        ),
        4,
    )
    # CHECK: both 5 4
    print("both", written(a), written(b))

    var appended = append_until(100)
    var misplaced = 0
    for i in range(len(appended)):
        if appended[i] != i + 1:
            misplaced += 1
    # CHECK: append 100 0
    print("append", len(appended), misplaced)
