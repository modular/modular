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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import UMT5Config, UMT5ConfigBase
from .umt5 import UMT5EncoderModel


def _prepare_state_dict(
    weights: Weights,
    target_dtype: DType | None = None,
) -> dict[str, WeightData]:
    """Convert Weights to a raw state dict, normalizing tied embedding keys.

    HF UMT5 ties ``shared.weight`` and ``encoder.embed_tokens.weight``.
    Our module owns the embedding as ``shared``, so we normalize to that key
    and drop the alias to avoid strict-mode validation failures.

    If ``target_dtype`` is provided, all weights are cast to that dtype
    (e.g. float32 → bfloat16 for Wan 2.1 checkpoints).
    """
    state_dict: dict[str, WeightData] = {}
    for key, value in weights.items():
        wd = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        state_dict[key] = wd

    encoder_emb = state_dict.pop("encoder.embed_tokens.weight", None)
    if "shared.weight" not in state_dict and encoder_emb is not None:
        state_dict["shared.weight"] = encoder_emb

    return state_dict


class UMT5Model(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.session = session or InferenceSession(devices=devices)
        self.config: UMT5ConfigBase = UMT5Config.generate(
            config,
            encoding,
            devices,
        )
        self.load_model()

    def load_model(self) -> Model:
        assert self.weights is not None, "Weights already freed"
        # Force bfloat16 — some repos (Wan 2.1) declare float32 but
        # should run in bfloat16 on GPU. Override both config and weights.
        dtype = DType.bfloat16
        self.config.dtype = dtype
        state_dict = _prepare_state_dict(self.weights, target_dtype=dtype)
        dev = self.devices[0]
        dev_ref = DeviceRef.from_device(dev)

        # Build module and load weights
        module = UMT5EncoderModel(self.config, dtype=dtype, device=dev_ref)
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Build graph with symbolic sequence length
        # attention_mask comes in as int64 from the pipeline
        input_types = [
            TensorType(DType.int64, ["batch", "seq_len"], device=dev),
            TensorType(DType.int64, ["batch", "seq_len"], device=dev),
        ]
        with Graph("umt5_encoder", input_types=input_types) as graph:
            input_ids = graph.inputs[0].tensor
            attention_mask = graph.inputs[1].tensor
            out = module(input_ids, attention_mask)
            graph.output(out)

        self.model: Model = self.session.load(
            graph, weights_registry=module.state_dict()
        )
        # Free raw weights after compilation
        self.weights = None  # type: ignore[assignment]
        return self.model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
