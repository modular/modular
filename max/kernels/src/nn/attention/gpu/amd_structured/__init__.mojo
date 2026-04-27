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
"""TileTensor-native attention kernels for AMD gfx950 (MI355X).

This module provides gfx950-only attention implementation using
TileTensor throughout. Supports MHA prefill (depth=64, 128, 256, 512),
MHA decode (token generation), MLA prefill, and MLA decode.
"""
