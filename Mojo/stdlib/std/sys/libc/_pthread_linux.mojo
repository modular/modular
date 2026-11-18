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

from std.ffi import (
    c_long,
    c_ulong,
    c_int,
    c_long_long,
    c_uint,
)
from std.sys.info import align_of, bit_width_of, CompilationTarget


@fieldwise_init
struct _pthread_sizes:
    var pthread_t: Int
    var attr_t: Int
    var mutexattr_t: Int
    var mtx_t: Int
    var condattr_t: Int
    var cond_t: Int
    var once_t: Int
    var rwlock_t: Int
    var rwlockattr_t: Int


def _linux_nptl_pthread_sizes() -> _pthread_sizes:
    comptime if CompilationTarget._is_triple_arch["aarch64"]():
        return _pthread_sizes(
            pthread_t=8,
            attr_t=64,
            mutexattr_t=8,
            mtx_t=48,
            condattr_t=8,
            cond_t=48,
            once_t=bit_width_of[c_int](),
            rwlock_t=56,
            rwlockattr_t=8,
        )
    elif CompilationTarget._is_triple_arch["x86_64"]():
        # TODO: This is for x86_64-linux-gnu
        # x86_64-linux-gnux32 is DIFFERENT and has different sizes
        # it's also different from regular i686-linux-gnu
        return _pthread_sizes(
            pthread_t=8,
            attr_t=56,
            mutexattr_t=4,
            mtx_t=40,
            condattr_t=4,
            cond_t=48,
            once_t=bit_width_of[c_int](),
            rwlock_t=56,
            rwlockattr_t=8,
        )
    else:
        comptime assert False


@align(align_of[c_long_long]())
struct pthread_cond_t:
    var __size: Array[Byte, _linux_nptl_pthread_sizes().cond_t]


@align(align_of[c_int]())
struct pthread_condattr_t:
    var __size: Array[Byte, _linux_nptl_pthread_sizes().condattr_t]


@align(align_of[c_long]())
struct pthread_mutex_t:
    var __size: Array[Byte, _linux_nptl_pthread_sizes().mtx_t]


@align(align_of[c_int]())
struct pthread_mutexattr_t:
    var __size: Array[Byte, _linux_nptl_pthread_sizes().mutexattr_t]


@align(align_of[c_long]())
struct pthread_rwlock_t:
    var __size: Array[Byte, _linux_nptl_pthread_sizes().rwlock_t]


@align(align_of[c_long]())
struct pthread_rwlockattr_t:
    var __size: Array[Byte, _linux_nptl_pthread_sizes().rwlockattr_t]
