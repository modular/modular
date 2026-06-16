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
"""SM100 (Blackwell) sparse multi-head attention (MSA) kernels.

Unified per-token prefill + decode fork (`msa_1q`). Kept as a sibling of the
`nn` package rather than nested inside it so the block-sparse MSA dispatch
surface stays self-contained.
"""
