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
"""FFI wrapper for POSIX thread library."""


from std.ffi import (
    external_call,
    c_long,
    c_char,
    c_ulong,
    c_int,
    c_long_long,
    c_uint,
    CStringSpan,
)
from std.collections.string import StringSlice, StaticString
from std.memory import ImmPointer, MaybeUninit
from std.sys.info import CompilationTarget, bit_width_of
from std.time.time import _CTimeSpec
from ._pthread_darwin import (
    pthread_cond_t as _darwin_opaque_pthread_cond_t,
    pthread_condattr_t as _darwin_opaque_pthread_condattr_t,
    pthread_mutex_t as _darwin_opaque_pthread_mutex_t,
    pthread_mutexattr_t as _darwin_opaque_pthread_mutexattr_t,
    pthread_rwlock_t as _darwin_opaque_pthread_rwlock_t,
    pthread_rwlockattr_t as _darwin_opaque_pthread_rwlockattr_t,
)
from ._pthread_linux import (
    pthread_cond_t as _linux_nptl_opaque_pthread_cond_t,
    pthread_condattr_t as _linux_nptl_opaque_pthread_condattr_t,
    pthread_mutex_t as _linux_nptl_opaque_pthread_mutex_t,
    pthread_mutexattr_t as _linux_nptl_opaque_pthread_mutexattr_t,
    pthread_rwlock_t as _linux_nptl_opaque_pthread_rwlock_t,
    pthread_rwlockattr_t as _linux_nptl_opaque_pthread_rwlockattr_t,
)

comptime pthread_cond_t = _darwin_opaque_pthread_cond_t if CompilationTarget.is_macos() else _linux_nptl_opaque_pthread_cond_t if CompilationTarget.is_linux() else NoneType

comptime pthread_condattr_t = _darwin_opaque_pthread_condattr_t if CompilationTarget.is_macos() else _linux_nptl_opaque_pthread_condattr_t if CompilationTarget.is_linux() else NoneType

comptime pthread_mutex_t = _darwin_opaque_pthread_mutex_t if CompilationTarget.is_macos() else _linux_nptl_opaque_pthread_mutex_t if CompilationTarget.is_linux() else NoneType

comptime pthread_mutexattr_t = _darwin_opaque_pthread_mutexattr_t if CompilationTarget.is_macos() else _linux_nptl_opaque_pthread_mutexattr_t if CompilationTarget.is_linux() else NoneType

comptime pthread_rwlock_t = _darwin_opaque_pthread_rwlock_t if CompilationTarget.is_macos() else _linux_nptl_opaque_pthread_rwlock_t if CompilationTarget.is_linux() else NoneType

comptime pthread_rwlockattr_t = _darwin_opaque_pthread_rwlockattr_t if CompilationTarget.is_macos() else _linux_nptl_opaque_pthread_rwlockattr_t if CompilationTarget.is_linux() else NoneType

# ===----------------------------------------------------------------------=== #
#     Codegen: Remaining symbols
# ===----------------------------------------------------------------------=== #


def pthread_rwlock_unlock(rwlock: MutPointer[pthread_rwlock_t, _]) -> c_int:
    return external_call["pthread_rwlock_unlock", c_int](rwlock)


def pthread_cond_init(
    cond: MutPointer[pthread_cond_t, _], attr: ImmPointer[pthread_condattr_t, _]
) -> c_int:
    return external_call["pthread_cond_init", c_int](cond, attr)


def pthread_cond_wait(
    cond: MutPointer[pthread_cond_t, _], mutex: MutPointer[pthread_mutex_t, _]
) -> c_int:
    return external_call["pthread_cond_wait", c_int](cond, mutex)


def pthread_mutexattr_destroy(
    mutexattr: MutPointer[pthread_mutexattr_t, _]
) -> c_int:
    return external_call["pthread_mutexattr_destroy", c_int](mutexattr)


def pthread_mutexattr_setprioceiling(
    mutexattr: MutPointer[pthread_mutexattr_t, _], prioceiling: c_int
) -> c_int:
    return external_call["pthread_mutexattr_setprioceiling", c_int](
        mutexattr, prioceiling
    )


def pthread_mutex_lock(mutex: MutPointer[pthread_mutex_t, _]) -> c_int:
    return external_call["pthread_mutex_lock", c_int](mutex)


def pthread_mutex_setprioceiling(
    mutex: MutPointer[pthread_mutex_t, _],
    policy: c_int,
    param: ImmPointer[c_int, _],
) -> c_int:
    return external_call["pthread_mutex_setprioceiling", c_int](
        mutex, policy, param
    )


def pthread_rwlock_trywrlock(rwlock: MutPointer[pthread_rwlock_t, _]) -> c_int:
    return external_call["pthread_rwlock_trywrlock", c_int](rwlock)


def pthread_mutex_trylock(mutex: MutPointer[pthread_mutex_t, _]) -> c_int:
    return external_call["pthread_mutex_trylock", c_int](mutex)


def pthread_rwlock_destroy(mutex: MutPointer[pthread_rwlock_t, _]) -> c_int:
    return external_call["pthread_rwlock_destroy", c_int](mutex)


def pthread_mutexattr_settype(
    mutexattr: MutPointer[pthread_mutexattr_t, _], type: c_int
) -> c_int:
    return external_call["pthread_mutexattr_settype", c_int](mutexattr, type)


def pthread_rwlockattr_destroy(
    attr: MutPointer[pthread_rwlockattr_t, _]
) -> c_int:
    return external_call["pthread_rwlockattr_destroy", c_int](attr)


def pthread_rwlock_wrlock(rwlock: MutPointer[pthread_rwlock_t, _]) -> c_int:
    return external_call["pthread_rwlock_wrlock", c_int](rwlock)


def pthread_condattr_init(attr: MutPointer[pthread_condattr_t, _]) -> c_int:
    return external_call["pthread_condattr_init", c_int](attr)


def pthread_mutex_unlock(mutex: MutPointer[pthread_mutex_t, _]) -> c_int:
    return external_call["pthread_mutex_unlock", c_int](mutex)


def pthread_rwlock_init(
    rwlock: MutPointer[pthread_rwlock_t, _],
    attr: ImmPointer[pthread_rwlockattr_t, _],
) -> c_int:
    return external_call["pthread_rwlock_init", c_int](rwlock, attr)


def pthread_rwlockattr_getpshared(
    attr: ImmPointer[pthread_rwlockattr_t, _], shared: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_rwlockattr_getpshared", c_int](attr, shared)


def pthread_mutexattr_getprioceiling(
    attr: ImmPointer[pthread_mutexattr_t, _], prioceiling: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_mutexattr_getprioceiling", c_int](
        attr, prioceiling
    )


def pthread_mutexattr_init(
    mutexattr: MutPointer[pthread_mutexattr_t, _]
) -> c_int:
    return external_call["pthread_mutexattr_init", c_int](mutexattr)


def pthread_mutexattr_getpshared(
    attr: ImmPointer[pthread_mutexattr_t, _], pshared: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_mutexattr_getpshared", c_int](attr, pshared)


def pthread_rwlockattr_setpshared(
    attr: MutPointer[pthread_rwlockattr_t, _], pshared: c_int
) -> c_int:
    return external_call["pthread_rwlockattr_setpshared", c_int](attr, pshared)


def pthread_mutexattr_setprotocol(
    attr: MutPointer[pthread_mutexattr_t, _], protocol: c_int
) -> c_int:
    return external_call["pthread_mutexattr_setprotocol", c_int](attr, protocol)


def pthread_condattr_destroy(attr: MutPointer[pthread_condattr_t, _]) -> c_int:
    return external_call["pthread_condattr_destroy", c_int](attr)


def pthread_mutexattr_getprotocol(
    attr: ImmPointer[pthread_mutexattr_t, _], protocol: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_mutexattr_getprotocol", c_int](attr, protocol)


def pthread_condattr_setpshared(
    condattr: MutPointer[pthread_condattr_t, _], pshared: c_int
) -> c_int:
    return external_call["pthread_condattr_setpshared", c_int](
        condattr, pshared
    )


def pthread_cond_destroy(cond: MutPointer[pthread_cond_t, _]) -> c_int:
    return external_call["pthread_cond_destroy", c_int](cond)


def pthread_mutex_init(
    mutex: MutPointer[pthread_mutex_t, _],
    mutexattr: ImmPointer[pthread_mutexattr_t, _],
) -> c_int:
    return external_call["pthread_mutex_init", c_int](mutex, mutexattr)


def pthread_rwlockattr_init(attr: MutPointer[pthread_rwlockattr_t, _]) -> c_int:
    return external_call["pthread_rwlockattr_init", c_int](attr)


def pthread_condattr_getpshared(
    attr: ImmPointer[pthread_condattr_t, _], pshared: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_condattr_getpshared", c_int](attr, pshared)


def pthread_cond_signal(cond: MutPointer[pthread_cond_t, _]) -> c_int:
    return external_call["pthread_cond_signal", c_int](cond)


def pthread_mutexattr_setpshared(
    attr: MutPointer[pthread_mutexattr_t, _], pshared: c_int
) -> c_int:
    return external_call["pthread_mutexattr_setpshared", c_int](attr, pshared)


def pthread_rwlock_tryrdlock(rwlock: ImmPointer[pthread_rwlock_t, _]) -> c_int:
    return external_call["pthread_rwlock_tryrdlock", c_int](rwlock)


def pthread_mutexattr_gettype(
    attr: ImmPointer[pthread_mutexattr_t, _], type: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_mutexattr_gettype", c_int](attr, type)


def pthread_mutex_destroy(mutex: MutPointer[pthread_mutex_t, _]) -> c_int:
    return external_call["pthread_mutex_destroy", c_int](mutex)


def pthread_mutex_getprioceiling(
    mutex: ImmPointer[pthread_mutex_t, _], prioceiling: MutPointer[c_int, _]
) -> c_int:
    return external_call["pthread_mutex_getprioceiling", c_int](
        mutex, prioceiling
    )


def pthread_rwlock_rdlock(rwlock: MutPointer[pthread_rwlock_t, _]) -> c_int:
    return external_call["pthread_rwlock_rdlock", c_int](rwlock)


def pthread_cond_timedwait(
    cond: MutPointer[pthread_cond_t, _],
    mutex: MutPointer[pthread_mutex_t, _],
    abstime: ImmPointer[_CTimeSpec, _],
) -> c_int:
    return external_call["pthread_cond_timedwait", c_int](cond, mutex, abstime)


def pthread_cond_broadcast(cond: MutPointer[pthread_cond_t, _]) -> c_int:
    return external_call["pthread_cond_broadcast", c_int](cond)
