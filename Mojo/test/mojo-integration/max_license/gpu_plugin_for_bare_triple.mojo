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
# RUN: kgen -emit=asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s --check-prefix=PTX
# RUN: kgen -emit=llvm --target-triple=amdgcn-amd-amdhsa --target-cpu=gfx942 --target-features="" %s | FileCheck %s --check-prefix=AMD

# A target named only by its triple carries no plugin payload, so it relies on
# the stdlib picking its vendor's plugin. Reading a GPU id needs that plugin,
# and `rsqrt` shows which plugin's lowering was used.

from std._gpu import thread_idx
from std.math import rsqrt


# PTX-LABEL: .visible .entry probe(
# PTX: mov.u32 {{%r[0-9]+}}, %tid.x;
# PTX: rsqrt.approx.ftz.f32
# AMD-LABEL: define {{.*}}amdgpu_kernel void @probe(
# AMD: call i32 @llvm.amdgcn.workitem.id.x()
# AMD: call float @llvm.amdgcn.rsq.f32(
@export
def probe(p: MutPointer[Float32, MutAnyOrigin]) abi("Mojo"):
    var i = thread_idx.x
    p[i] = rsqrt(p[i])
