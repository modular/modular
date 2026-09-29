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
"""Mamba2 state-space architecture (SSD chunk-scan path).

This package is the Python counterpart of the Mojo
``max/kernels/src/state_space/ssd_*`` kernels registered under the
``ssd_chunk_scan_combined`` op name. The first deliverable exposes the
functional-op wrapper; NN modules and the full pipeline land in later
RFC 0003 increments.
"""

from .functional_ops import ssd_chunk_scan_combined

__all__ = [
    "ssd_chunk_scan_combined",
]
