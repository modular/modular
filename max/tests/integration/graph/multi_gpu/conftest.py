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
from collections.abc import Generator

import pytest
from max.driver import Accelerator, accelerator_count


@pytest.fixture(autouse=True)
def clean_up_gpus() -> Generator[None, None, None]:
    """Call synchronize after each test on all accelerators.

    GPU failures for a particular device can spill over to later tests,
    incorrectly reporting the source of the error. This fixture synchronizes
    all accelerators after each test, which will propagate any pending errors
    up to the Python level.
    """

    yield

    for i in range(accelerator_count()):
        accelerator = Accelerator(i)
        accelerator.synchronize()
