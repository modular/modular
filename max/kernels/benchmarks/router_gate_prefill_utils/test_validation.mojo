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

from std.math import inf, nan
from std.memory import bitcast
from std.pathlib import Path
from std.tempfile import NamedTemporaryFile
from std.testing import assert_equal, assert_raises

from router_gate_prefill_utils.validation import max_abs_diff, read_f32


def test_max_abs_diff() raises:
    var a = List[Float32](length=2, fill=1)
    var b = List[Float32](length=2, fill=2)
    assert_equal(max_abs_diff(Span(a), Span(b), 2), 1)
    for value in [nan[.float32](), inf[.float32](), -inf[.float32]()]:
        b[1] = value
        with assert_raises(contains="non-finite"):
            _ = max_abs_diff(Span(a), Span(b), 2)
        with assert_raises(contains="non-finite"):
            _ = max_abs_diff(Span(b), Span(a), 2)


def test_read_f32() raises:
    with NamedTemporaryFile() as tmp:
        var path = Path(tmp.name)
        path.write_bytes("abcd".as_bytes())
        var values = read_f32(tmp.name, 1)
        assert_equal(len(values), 1)
        assert_equal(bitcast[.uint32](values[0]), UInt32(0x64636261))
        for count in [0, 2]:
            with assert_raises(contains="expected"):
                _ = read_f32(tmp.name, count)
        path.write_bytes("abc".as_bytes())
        with assert_raises(contains="expected"):
            _ = read_f32(tmp.name, 1)


def main() raises:
    test_max_abs_diff()
    test_read_f32()
