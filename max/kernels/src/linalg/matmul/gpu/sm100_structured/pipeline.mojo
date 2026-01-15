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

"""Re-export ProducerConsumerPipeline from legacy sm100 module.

This module previously duplicated the ProducerConsumerPipeline definition.
Now it re-exports from the canonical location in sm100.pipeline to ensure
type compatibility between structured and legacy code paths.

The legacy module's ProducerConsumerPipeline is structurally identical and
can be used seamlessly with all structured pipeline patterns (OutputStage,
TilePipeline, etc.).
"""

# Re-export ProducerConsumerPipeline and MbarPtr from the canonical location
from linalg.matmul.gpu.sm100.pipeline import (
    ProducerConsumerPipeline,
    MbarPtr,
)
