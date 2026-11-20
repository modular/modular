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
"""Compatibility import for the role-registered Muse Glimmer pipeline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.pipelines.lib.module_registration import (
        ModuleV3DecoderPipelineModel as MuseGlimmerModel,
    )
else:
    from .arch import muse_glimmer_arch

    MuseGlimmerModel = muse_glimmer_arch.pipeline_model
    assert MuseGlimmerModel is not None

__all__ = ["MuseGlimmerModel"]
