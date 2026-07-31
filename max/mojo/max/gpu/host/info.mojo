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
"""Re-exports accelerator target information from `std.gpu.host.info`."""

from std.gpu.host.info import *
from std.gpu.host.info import (
    _all_targets,
    _get_a100_target,
    _get_h100_target,
    _get_metal_m1_target,
    _get_metal_m2_target,
    _get_mi300x_target,
    _get_mi355x_target,
    _is_sm10x_gpu,
    _is_sm12x_gpu,
)
