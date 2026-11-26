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
# The page's `report.py`, verbatim.

import mojo.importer  # noqa: F401

# isort: split

from fast_stats import RunningStats  # type: ignore[import-not-found]

latency = RunningStats()
latency.add(12.5, 9.8, 14.1)
latency.add(11.0)

batch = RunningStats.from_values([30.2, 10.7])
latency.merge(batch)

print(f"mean {latency.mean():.2f} ms, stddev {latency.stddev():.2f} ms")
print(latency)
