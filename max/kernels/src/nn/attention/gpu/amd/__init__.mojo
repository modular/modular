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
"""AMD CDNA GPU attention kernels for GFX942/GFX950 architectures.

Includes MHA prefill/decode, MLA, matrix-multiply-accumulate primitives,
shared-memory buffers, and softmax helpers. RDNA kernels are in amd_rdna/.
"""
