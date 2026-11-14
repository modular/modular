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
"""Names Xiaomi's MiMo-V2.6-Flash checkpoint tensors as MAX's weights.

Every tensor loads as the checkpoint stores it; the model reads Xiaomi's
layouts itself. The vision, audio and MTP tensors are not loaded.

The repo also ships the audio tokenizer and the DFlash drafter as
safetensors beside the text model's shards, which weight discovery globs
in; the text model's tensors are those of its safetensors index, which
:func:`text_model_weights` filters the rest out by.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import HuggingFaceRepo, PipelineConfig

_IGNORED_PREFIXES = (
    "visual.",
    "audio_encoder.",
    "speech_embeddings.",
    "model.mtp.",
)

_WEIGHT_INDEX = "model.safetensors.index.json"


def text_model_weights(
    repo: HuggingFaceRepo, weights: Weights | Mapping[str, Weights]
) -> dict[str, Weights]:
    """Returns the weights the text model's safetensors index lists.

    The repo also ships the audio tokenizer and the DFlash drafter as
    safetensors in subdirectories, which weight discovery globs in. The text
    model's tensors are those of the index in the model's folder (the repo's
    ``subfolder``, or its root); a checkpoint without an index is a single
    file, all of it the model's.

    Args:
        repo: The model's repo.
        weights: Every tensor weight discovery found.

    Returns:
        The text model's tensors, by checkpoint name.
    """
    index = (
        f"{repo.subfolder}/{_WEIGHT_INDEX}" if repo.subfolder else _WEIGHT_INDEX
    )
    if repo.repo_type == "local":
        path = os.path.join(repo.local_path, index)
        if not os.path.isfile(path):
            return dict(weights.items())
    else:
        try:
            path = huggingface_hub.hf_hub_download(
                repo.repo_id, index, revision=repo.revision
            )
        except EntryNotFoundError:
            return dict(weights.items())
    with open(path) as f:
        names = set(json.load(f)["weight_map"])
    return {name: w for name, w in weights.items() if name in names}


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Renames the checkpoint's text-model tensors to MAX's MiMo-V2 weights.

    Args:
        state_dict: Every tensor in the checkpoint shards, by checkpoint name.
        **unused_kwargs: Accepted so every adapter shares one signature.

    Returns:
        The weights keyed by MAX name: the checkpoint name without
        ``model.``, with each FP8 projection's ``weight_scale_inv`` as
        ``weight_scale`` and the router's ``mlp.gate.weight`` as
        ``mlp.gate.gate_score.weight``. The experts' ``weight_scale`` bytes
        are E8M0, which the checkpoint stores as ``U8``.
    """
    adapted: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        if name.startswith(_IGNORED_PREFIXES):
            continue
        data = weight.data()
        max_name = name.removeprefix("model.")
        if max_name.endswith(".weight_scale_inv"):
            max_name = max_name.removesuffix("_inv")
        elif max_name.endswith(".mlp.gate.weight"):
            max_name = max_name.replace(".gate.", ".gate.gate_score.")
        elif ".mlp.experts." in max_name and max_name.endswith(".weight_scale"):
            # ModuleV3's loader checks the backing array's dtype, not the
            # field below, so retype the buffer itself. E8M0 is a
            # byte-for-byte reinterpretation of the checkpoint's U8 scales.
            if data.dtype == DType.uint8:
                scales = data.to_buffer().view(
                    DType.float8_e8m0fnu, [int(dim) for dim in data.shape]
                )
                data = dataclasses.replace(
                    data, data=scales, dtype=DType.float8_e8m0fnu
                )
        adapted[max_name] = dataclasses.replace(data, name=max_name)
    return adapted


def registered_state_dict(
    state_dict: Mapping[str, Weights],
    *,
    pipeline_config: PipelineConfig,
    **kwargs: object,
) -> dict[str, WeightData]:
    """Adapts only the text checkpoint index's tensors for role registration.

    The repo also ships the audio tokenizer and the DFlash drafter as
    safetensors beside the text model's shards, and weight discovery globs
    them all in: the drafter's bare ``layers.0.*`` names collide with the
    decoder's own, so the text model's index filters them out before the
    adapter renames anything.
    """
    return convert_safetensor_state_dict(
        text_model_weights(
            pipeline_config.model.huggingface_weight_repo, state_dict
        ),
        **kwargs,
    )
