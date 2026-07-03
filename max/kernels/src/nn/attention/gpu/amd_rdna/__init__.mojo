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
"""TileTensor-native attention kernels for AMD RDNA3+ (gfx11xx/gfx12xx).

Wave32 with 16x16x16 WMMA. 16-element A/B fragments per lane (full K),
8-element C/D fragments per lane. Supports MHA prefill and decode.
"""
