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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import pytest
from max.pipelines.architectures.inkling.model_config import (
    parse_inkling_mtp_config,
)
from max.pipelines.architectures.mtp_inkling.weight_adapters import (
    inkling_mtp_weight_rules,
)
from max.pipelines.lib.weight_transform import WeightTransform


def test_parse_inkling_mtp_config_reads_namespace_and_caps_depths() -> None:
    hf = SimpleNamespace(
        mtp_config=SimpleNamespace(
            num_nextn_predict_layers=8,
            chain_hidden_post_norm=True,
            local_layer_ids=[0, 2, 4],
            mtp_hidden_states_first=False,
        )
    )
    mtp = parse_inkling_mtp_config(hf)
    assert mtp is not None
    assert mtp.num_nextn_predict_layers == 8
    assert mtp.chain_hidden_post_norm is True
    assert mtp.local_layer_ids == (0, 2, 4)
    assert mtp.hidden_states_first is False
    assert mtp.num_depths(2) == 2
    assert mtp.num_depths(16) == 8


def test_parse_inkling_mtp_config_reads_dict_and_defaults() -> None:
    hf = SimpleNamespace(
        mtp_config={
            "num_nextn_predict_layers": 4,
            "local_layer_ids": [1],
        }
    )
    mtp = parse_inkling_mtp_config(hf)
    assert mtp is not None
    assert mtp.num_depths(3) == 3
    assert mtp.chain_hidden_post_norm is False
    assert mtp.local_layer_ids == (1,)
    assert mtp.hidden_states_first is True


def test_parse_inkling_mtp_config_missing_or_empty_is_none() -> None:
    assert parse_inkling_mtp_config(SimpleNamespace()) is None
    assert (
        parse_inkling_mtp_config(
            SimpleNamespace(mtp_config={"num_nextn_predict_layers": 0})
        )
        is None
    )


def test_inkling_mtp_config_num_depths_rejects_empty_spec() -> None:
    hf = SimpleNamespace(mtp_config={"num_nextn_predict_layers": 2})
    mtp = parse_inkling_mtp_config(hf)
    assert mtp is not None
    with pytest.raises(ValueError, match="at least one speculative token"):
        mtp.num_depths(0)


def test_inkling_mtp_rules_keep_only_the_depths() -> None:
    weight = NonCallableMock()
    state_dict = {
        "model.llm.embed.weight": weight,
        "model.llm.layers.0.attn.wq_du.weight": weight,
        "model.mtp.chain_norm.weight": weight,
        "model.mtp.layers.0.hidden_norm.weight": weight,
        "model.mtp.layers.0.embed_norm.weight": weight,
        "model.mtp.layers.0.input_proj.weight": weight,
        "model.mtp.layers.0.transformer_block.attn.wq_du.weight": weight,
        "model.mtp.layers.0.transformer_block.attn_norm.weight": weight,
        "model.audio.unused.weight": weight,
        "model.visual.unused.weight": weight,
    }

    new_state_dict = WeightTransform(
        inkling_mtp_weight_rules(NonCallableMock())
    )(state_dict)

    assert set(new_state_dict) == {
        "chain_norm.weight",
        "layers.0.hidden_norm.weight",
        "layers.0.embed_norm.weight",
        "layers.0.input_proj.weight",
        "layers.0.decoder_layer.attn.wq_du.weight",
        "layers.0.decoder_layer.attn_norm.weight",
    }


def _depth_weight_names(depth: int) -> list[str]:
    return [
        f"model.mtp.layers.{depth}.hidden_norm.weight",
        f"model.mtp.layers.{depth}.input_proj.weight",
        f"model.mtp.layers.{depth}.transformer_block.attn.wq_du.weight",
    ]


def test_inkling_mtp_rules_keep_checkpoint_depths_without_reading_metadata() -> (
    None
):
    state_dict = {
        name: NonCallableMock()
        for depth in range(8)
        for name in _depth_weight_names(depth)
    }
    ignored = NonCallableMock()
    state_dict[
        "model.mtp.layers.0.transformer_block.mlp.w13_dn.original_shape"
    ] = ignored
    converted = WeightTransform(inkling_mtp_weight_rules(NonCallableMock()))(
        state_dict
    )
    for depth in range(8):
        assert f"layers.{depth}.hidden_norm.weight" in converted
    ignored.data.assert_not_called()
