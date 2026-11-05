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

"""Normalization kernels as ``Tensor`` ops."""

from max import functional as F
from max.nn._graph_ops import mtp_eh_norm as _mtp_eh_norm

mtp_eh_norm = F.functional(_mtp_eh_norm)


__all__ = [
    "mtp_eh_norm",
]
