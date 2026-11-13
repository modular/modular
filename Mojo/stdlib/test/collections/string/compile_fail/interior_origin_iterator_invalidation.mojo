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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Iterators and slices vended by a `String` carry an interior origin, so
# mutating the string (which may reallocate its buffer) invalidates them at
# compile time.


def bytes_iter():
    var s = String("hello world")
    var it = s.bytes()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for b in it:
        print(b)


def codepoints_iter():
    var s = String("hello world")
    var it = s.codepoints()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for c in it:
        print(c)


def codepoint_slices_iter():
    var s = String("hello world")
    var it = s.codepoint_slices()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for c in it:
        print(c)


def codepoint_slices_reversed_iter():
    var s = String("hello world")
    var it = s.codepoint_slices_reversed()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for c in it:
        print(c)


def graphemes_iter():
    var s = String("hello world")
    var it = s.graphemes()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for g in it:
        print(g)


def graphemes_reversed_iter():
    var s = String("hello world")
    var it = s.graphemes_reversed()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for g in it:
        print(g)


def grapheme_indices_iter():
    var s = String("hello world")
    var it = s.grapheme_indices()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for i, g in it:
        print(i, g)


def string_iter():
    var s = String("hello world")
    var it = s.__iter__()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for g in it:
        print(g)


def string_reversed():
    var s = String("hello world")
    var it = s.__reversed__()
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    for g in it:
        print(g)


def split_at_grapheme_halves():
    var s = String("hello world")
    var halves = s.split_at_grapheme(5)
    s += "some more text to force a reallocation"
    # CHECK: [[@LINE+1]]:{{[0-9]+}}: error: use of invalidated interior reference 's["bytes"]'
    print(halves[1])


def main():
    bytes_iter()
    codepoints_iter()
    codepoint_slices_iter()
    codepoint_slices_reversed_iter()
    graphemes_iter()
    graphemes_reversed_iter()
    grapheme_indices_iter()
    string_iter()
    string_reversed()
    split_at_grapheme_halves()
