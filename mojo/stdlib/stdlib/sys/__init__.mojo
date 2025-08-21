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
"""Implements the sys package."""

from ._io import stderr, stdin, stdout
from .arg import argv
from .compile import is_compile_time
from .debug import breakpointhook
from .ffi import DEFAULT_RTLD, RTLD, DLHandle, external_call
from .info import (
    CompilationTarget,
    alignof,
    bitwidthof,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_big_endian,
    is_apple_gpu,
    is_gpu,
    is_little_endian,
    is_nvidia_gpu,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    simdbitwidth,
    simdbytewidth,
    simdwidthof,
    sizeof,
)
from .intrinsics import (
    PrefetchCache,
    PrefetchLocality,
    PrefetchOptions,
    PrefetchRW,
    _RegisterPackType,
    compressed_store,
    gather,
    llvm_intrinsic,
    masked_load,
    masked_store,
    prefetch,
    scatter,
    strided_load,
    strided_store,
)
from .param_env import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    env_get_string,
    is_defined,
)
from .terminate import exit
