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
"""Top-level pipeline dispatch helpers for cascade.

This is a temporary string-matching shim until cascade pipeline selection can
integrate with the MAX architectures registry. For now it only knows how to
build the in-process dummy pipelines used by the experimental CLI and tests.
"""

from __future__ import annotations

from max.experimental.cascade.pipelines import CascadePipeline
from max.experimental.cascade.pipelines.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)
from max.pipelines.lib.config import PipelineConfig


async def build_pipeline(
    config: PipelineConfig,
) -> CascadePipeline:
    """Build the cascade pipeline described by *config*.

    Routes to the correct builder based on the main ``model_path``.

    Args:
        config: Fully-specified ``PipelineConfig``, typically constructed
            by cyclopts from ``--models.*`` CLI flags.
    """
    model_path = config.model.model_path

    if model_path == "dummy_textgen":
        return await build_dummy_textgen_pipeline()

    if model_path == "dummy_imgen":
        return await build_dummy_imgen_pipeline()

    raise ValueError(
        f"Unsupported model {model_path!r}. "
        "cascade currently supports the dummy_textgen and dummy_imgen "
        "pipelines."
    )
