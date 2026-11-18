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

from std.ffi import c_ulong, c_int, c_long
from std.sys.info import bit_width_of, CompilationTarget


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


def _darwin_pthread_sizes() -> _pthread_sizes:
    comptime match bit_width_of[DType.uint]():
        case 64:
            return _pthread_sizes(
                pthread_t=8176,
                attr_t=56,
                mutexattr_t=8,
                mtx_t=56,
                condattr_t=8,
                cond_t=40,
                once_t=8,
                rwlock_t=192,
                rwlockattr_t=16,
            )
        case 32:
            return _pthread_sizes(
                pthread_t=4088,
                attr_t=36,
                mutexattr_t=8,
                mtx_t=40,
                condattr_t=4,
                cond_t=24,
                once_t=4,
                rwlock_t=124,
                rwlockattr_t=12,
            )
        case _:
            comptime assert False


struct _opaque_pthread_cond_t:
    var __sig: c_long
    var __opaque: Array[Byte, _darwin_pthread_sizes().cond_t]


struct _opaque_pthread_condattr_t:
    var __sig: c_long
    var __opaque: Array[Byte, _darwin_pthread_sizes().condattr_t]


struct _opaque_pthread_mutex_t:
    var __sig: c_long
    var __opaque: Array[Byte, _darwin_pthread_sizes().mtx_t]


struct _opaque_pthread_mutexattr_t:
    var __sig: c_long
    var __opaque: Array[Byte, _darwin_pthread_sizes().mutexattr_t]


struct _opaque_pthread_rwlock_t:
    var __sig: c_long
    var __opaque: Array[Byte, _darwin_pthread_sizes().rwlock_t]


struct _opaque_pthread_rwlockattr_t:
    var __sig: c_long
    var __opaque: Array[Byte, _darwin_pthread_sizes().rwlockattr_t]


comptime pthread_cond_t = _opaque_pthread_cond_t
comptime pthread_condattr_t = _opaque_pthread_condattr_t
comptime pthread_mutex_t = _opaque_pthread_mutex_t
comptime pthread_mutexattr_t = _opaque_pthread_mutexattr_t
comptime pthread_rwlock_t = _opaque_pthread_rwlock_t
comptime pthread_rwlockattr_t = _opaque_pthread_rwlockattr_t
