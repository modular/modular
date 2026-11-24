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
"""Opaque CUDA and HIP handle types.

The handle structs and their aliases are the same for the legacy and the HAL
backend. The accessors that read them from a context, stream or function stay
with each backend, because they take that backend's own type.
"""

from std.memory import Pointer

# ===-------------------------------------------------------------------===#
# CUDA
# ===-------------------------------------------------------------------===#


struct _CUctx_st:
    pass


struct _CUstream_st:
    pass


struct _CUmod_st:
    pass


struct _CUevent_st:
    pass


comptime CUcontext = Optional[Pointer[_CUctx_st, MutUntrackedOrigin]]
comptime CUstream = Optional[Pointer[_CUstream_st, MutUntrackedOrigin]]
comptime CUmodule = Optional[Pointer[_CUmod_st, MutUntrackedOrigin]]
comptime CUevent = Optional[Pointer[_CUevent_st, MutUntrackedOrigin]]

# ===-------------------------------------------------------------------===#
# HIP
# ===-------------------------------------------------------------------===#


struct _ihipDevice_t:
    pass


struct _ihipStream_t:
    pass


struct _ihipModule_t:
    pass


comptime hipDevice_t = Optional[Pointer[_ihipDevice_t, MutUntrackedOrigin]]
comptime hipStream_t = Optional[Pointer[_ihipStream_t, MutUntrackedOrigin]]
comptime hipModule_t = Optional[Pointer[_ihipModule_t, MutUntrackedOrigin]]
