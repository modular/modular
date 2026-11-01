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
#
# RUN: mojo doc %s | FileCheck %s


from max.gpu.host.info import GPUInfo, AMDRDNAFamily

# CHECK: "name": "Radeon7600",
# CHECK: "value": "GPUInfo.from_family(AMDRDNAFamily, StringSpan(\"Radeon 7600\"), StringSpan(\"hip\"), StringSpan(\"gfx1102\"), SIMD(11), StringSpan(\"RDNA3\"), Int(32))"
comptime Radeon7600 = GPUInfo.from_family(
    family=AMDRDNAFamily,
    name="Radeon 7600",
    api="hip",
    arch_name="gfx1102",
    compute=11.0,
    version="RDNA3",
    sm_count=32,
)
