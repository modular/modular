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

from __future__ import annotations

import time

from max.profiler.gpu import BackgroundRecorder


def test_smoke() -> None:
    with BackgroundRecorder() as recorder1:
        time.sleep(3)
    assert len(recorder1.stats) > 0

    with BackgroundRecorder(interval=0.1) as recorder2:
        time.sleep(3)
    assert len(recorder2.stats) > len(recorder1.stats)
