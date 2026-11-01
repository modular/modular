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
"""Compatibility import for the framework-generated glm5_next_modulev3 pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.driver import Device, accelerator_api
from max.dtype import DType
from typing_extensions import override

from .checkpoint import _hc_fn_dtype as _checkpoint_hc_fn_dtype
from .checkpoint import _stored_hc_fn_dtype, cast_hc_fn_weights

if TYPE_CHECKING:
    from max.pipelines.lib.module_registration import (
        ModuleV3DecoderPipelineModel as _RegisteredGlm5NextModel,
    )
else:
    from .arch import glm5_next_modulev3_arch

    _RegisteredGlm5NextModel = glm5_next_modulev3_arch.pipeline_model
    assert _RegisteredGlm5NextModel is not None


def _hc_fn_dtype(
    stored: DType, hc_mult: int, hidden_size: int, dtype: DType, device: Device
) -> DType:
    """Resolves mHC precision through the legacy module's device API hook."""
    return _checkpoint_hc_fn_dtype(
        stored, hc_mult, hidden_size, dtype, device, api=accelerator_api()
    )


class Glm5NextModel(_RegisteredGlm5NextModel):
    """Preserves legacy weight preparation for callers of the old model class.

    Architecture registration uses the generated pipeline directly. This
    compatibility class delegates precision handling to the same helper as
    the registered checkpoint adapter.
    """

    @override
    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        return cast_hc_fn_weights(
            super()._prepare_state_dict(state_dict, model_config),
            model_config.hc_fn_dtype,
        )


__all__ = ["Glm5NextModel", "_hc_fn_dtype", "_stored_hc_fn_dtype"]
