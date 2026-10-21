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
"""How many accelerator slots the eager-GC warm compiles."""

# Per-slot artifacts make the warm device-count-independent, so this is a
# ceiling, not an exact match: a box with <= this many GPUs loads slots 0..k-1.
# Set to the largest GPU count CI serves (the has_4_gpus tier, there is no
# has_8_gpus), so any smaller box is covered by one warm; a larger box fails,
# as it has no artifact for its extra slots.
WARM_DEVICE_COUNT = 4
