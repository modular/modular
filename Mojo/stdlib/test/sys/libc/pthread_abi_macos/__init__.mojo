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
from std.sys.libc._pthread_darwin import (
    pthread_cond_t,
    pthread_condattr_t,
    pthread_mutex_t,
    pthread_mutexattr_t,
    pthread_rwlock_t,
    pthread_rwlockattr_t,
)
from test_utils.cabi_check import AbiConstant, AbiTypedefLike, AbiTypedef

comptime CABI_STRUCTS = TypeList.of[
    Trait=AnyType,
    pthread_cond_t,
    pthread_condattr_t,
    pthread_mutex_t,
    pthread_mutexattr_t,
    pthread_rwlock_t,
    pthread_rwlockattr_t,
]()

comptime CABI_CONSTANTS: List[AbiConstant] = []
comptime CABI_TYPEDEFS = TypeList.of[Trait=AbiTypedefLike,]()

comptime CABI_INCLUDES: List[StaticString] = ["<pthread.h>"]
