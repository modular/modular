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
"""Compatibility import for the framework-generated idefics3_modulev3 pipeline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.pipelines.lib.module_registration import (
        ModuleV3DecoderPipelineModel as Idefics3Model,
    )
else:
    from .arch import idefics3_modulev3_arch

    Idefics3Model = idefics3_modulev3_arch.pipeline_model
    assert Idefics3Model is not None

__all__ = ["Idefics3Model"]
