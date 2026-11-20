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
"""The pipeline model's compile inputs are the frozen ABI, slot for slot.

``DeepseekV41Model._get_compile_input_types`` asks the batch processor for
its inputs, so the graph the engine serves must compile from the same ABI
the stub records. Each TP degree stages the stub through the pipeline's
own path -- the model shell, the batch processor and the KV params -- and
the result must equal the recorded contract.
"""

from __future__ import annotations

import json

import abi_contract
import pipeline_inputs
import pytest


@pytest.mark.parametrize("tp", abi_contract.TP_DEGREES)
def test_the_pipeline_inputs_stage_to_the_recorded_contract(tp: int) -> None:
    with abi_contract.virtual_devices(tp):
        staged, config, kv_params = pipeline_inputs.stage_pipeline("full", tp)
        records = abi_contract.contract(staged, config, kv_params, tp)
    recorded = json.loads(
        (abi_contract.testdata_dir() / "abi_contract.json").read_text()
    )["layouts"][f"full_tp{tp}"]
    assert records == recorded
    # The three fixed inputs lead the ABI: no host context-length input may
    # sit between them and the first KV leaf.
    assert [r["name"] for r in records[:3]] == [
        "tokens",
        "return_n_logits",
        "input_row_offsets",
    ]
