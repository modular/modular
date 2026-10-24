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
"""The MTP depths an Inkling checkpoint ships beside its trunk."""

from .arch import mtp_inkling_draft_arch
from .model import MTPInklingDraftArchConfig
from .weight_adapters import convert_inkling_mtp_state_dict

__all__ = [
    "MTPInklingDraftArchConfig",
    "convert_inkling_mtp_state_dict",
    "mtp_inkling_draft_arch",
]
