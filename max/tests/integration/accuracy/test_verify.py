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

import json
from pathlib import Path

import numpy as np
from test_common.numpy_encoder import NumpyEncoder
from verify import verify


def _write_logit_output(path: Path, logits: np.ndarray) -> None:
    output = [
        {
            "prompt": "prompt",
            "values": [
                {
                    "next_token": 1,
                    "next_token_logits": float(logits[1]),
                    "logits": logits,
                }
            ],
        }
    ]
    path.write_text(json.dumps(output, cls=NumpyEncoder))


def test_verify_fails_when_logits_include_nan(tmp_path: Path) -> None:
    pipeline_outputs = tmp_path / "pipeline.json"
    torch_outputs = tmp_path / "torch.json"

    _write_logit_output(
        pipeline_outputs,
        np.array([np.nan, 0.5, -0.5], dtype=np.float32),
    )
    _write_logit_output(
        torch_outputs,
        np.array([1.0, 0.5, -0.5], dtype=np.float32),
    )

    result = verify(
        pipeline_outputs=pipeline_outputs,
        torch_outputs=torch_outputs,
        eval_metric=["cos", "kl"],
        cos_dist_threshold=1.0,
        kl_div_threshold=1.0,
    )

    assert not result.passed
    assert result.error_message is not None
