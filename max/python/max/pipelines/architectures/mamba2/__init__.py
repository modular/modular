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
"""Mamba2 state-space architecture (SSD chunk-scan path).

This package is the Python counterpart of the Mojo
``max/kernels/src/state_space/ssd_*`` kernels registered under the
``ssd_chunk_scan_combined`` op name. RFC 0003 item 5 wires the full
prefill/step pipeline:

* :func:`ssd_chunk_scan_combined` — functional-op wrapper.
* :class:`Mamba2Mixer` / :class:`Mamba2Block` — standalone NN modules.
* :class:`Mamba2Prefill` / :class:`Mamba2Step` — compile-time MAX-graph
  Modules consumed by the pipeline model.
* :class:`Mamba2SSMStateCache` — per-slot conv/SSM state cache (mirrors
  the Mamba1 :class:`SSMStateCache`).
* :class:`Mamba2Model` / :class:`Mamba2ModelInputs` — the pipeline
  model with prefill/step dispatch.
"""

from .arch import mamba2_arch
from .functional_ops import ssd_chunk_scan_combined
from .mamba2 import Mamba2Block, Mamba2Mixer, mamba2_dims
from .mamba2_module import Mamba2Prefill, Mamba2Step
from .model import Mamba2Model, Mamba2ModelInputs
from .model_config import Mamba2Config
from .ssm_cache import Mamba2SSMStateCache
from .weight_adapters import convert_mamba2_state_dict

__all__ = [
    "Mamba2Block",
    "Mamba2Config",
    "Mamba2Mixer",
    "Mamba2Model",
    "Mamba2ModelInputs",
    "Mamba2Prefill",
    "Mamba2SSMStateCache",
    "Mamba2Step",
    "convert_mamba2_state_dict",
    "mamba2_arch",
    "mamba2_dims",
    "ssd_chunk_scan_combined",
]
