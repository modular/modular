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

from sys import bitwidthof

from builtin.dtype import _integral_type_of
from memory import bitcast


fn ulp_distance[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Int:
    alias bitwidth = bitwidthof[dtype]()
    alias T = _integral_type_of[dtype]()
    # widen to Int first to avoid overflow
    var a_int = Int(bitcast[T](a))
    var b_int = Int(bitcast[T](b))
    # to twos complement
    alias two_complement_const = Int(1 << (bitwidth - 1))
    a_int = two_complement_const - a_int if a_int < 0 else a_int
    b_int = two_complement_const - b_int if b_int < 0 else b_int
    return abs(a_int - b_int)
