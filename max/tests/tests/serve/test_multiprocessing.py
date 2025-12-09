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


from multiprocessing.reduction import ForkingPickler

import numpy
from max.interfaces import RequestID, TextGenerationRequest


def test_reductions() -> None:
    # No extra reductions to register at the moment.

    request = TextGenerationRequest(
        request_id=RequestID("0"), prompt="test", model_name="test"
    )
    context = {
        "0": numpy.ones((3, 3), dtype=numpy.float32),
    }
    for obj in (request, context):
        assert ForkingPickler.dumps(obj)
