# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import pytest
from hypothesis import settings
from max.driver import CPU
from max.engine import InferenceSession

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multiple times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])
