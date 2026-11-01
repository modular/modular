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

"""Memory planner for the Muse Spark architecture."""

from __future__ import annotations

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner


class MuseSparkMemoryPlanner(PagedMemoryPlanner):
    """Reserves a fixed activation budget before the KV cache is sized."""

    # TODO: measured on one B200 with the 8-layer fixture at --max-length
    # 8192; remeasure for the 62-layer checkpoint, multi-GPU and vision.
    _activation_reservation_bytes = 16 * 1024**3
