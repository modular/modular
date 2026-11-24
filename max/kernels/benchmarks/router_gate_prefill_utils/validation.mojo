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

from std.math import isfinite
from std.pathlib import Path
from std.sys import size_of


def max_abs_diff(
    a: Span[Scalar[DType.float32], _], b: Span[Scalar[DType.float32], _], n: Int
) raises -> Float32:
    var d = Float32(0)
    for i in range(n):
        if not isfinite(a[i]) or not isfinite(b[i]):
            raise Error("non-finite benchmark output")
        d = max(d, abs(a[i] - b[i]))
    return d


def read_f32(path: String, count: Int) raises -> List[Float32]:
    var raw = Path(path).read_bytes()
    var expected_bytes = count * size_of[Float32]()
    if len(raw) != expected_bytes:
        raise Error(
            String(
                "expected ",
                expected_bytes,
                " bytes in ",
                path,
                ", got ",
                len(raw),
            )
        )
    var out = List[Float32](length=count, fill=0)
    var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(count):
        out[i] = src[unsafe_offset=i]
    return out^
