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
"""Tests for the rule-based weight transform."""

from __future__ import annotations

import json
import logging
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import Weights, load_weights
from max.pipelines.lib import (
    Bitcast,
    CheckpointRole,
    Ignore,
    WeightEntry,
    WeightNaming,
    WeightRenaming,
    WeightTransform,
    apply_weight_entries,
)

_SAFETENSORS_DTYPES = {np.dtype(np.float32): "F32", np.dtype(np.uint8): "U8"}


def _checkpoint(
    path: Path, tensors: Mapping[str, npt.NDArray[Any]]
) -> dict[str, Weights]:
    header: dict[str, object] = {}
    offset = 0
    for name, array in tensors.items():
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    blob = json.dumps(header).encode()
    # Pad the header so every tensor starts aligned.
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for array in tensors.values():
            f.write(array.tobytes())
    return dict(load_weights([path]).items())


def _numpy(buffer: object) -> npt.NDArray[Any]:
    assert isinstance(buffer, Buffer)
    return buffer.view(DType.uint8).to_numpy()


def test_renames_apply_in_order_to_every_match(tmp_path: Path) -> None:
    weights = _checkpoint(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.w.w": np.ones(2, np.float32),
            "model.norm": np.zeros(2, np.float32),
        },
    )
    transform = WeightTransform(
        [
            WeightRenaming(r"^model\.", ""),
            WeightRenaming(r"w", "proj"),
            WeightRenaming(r"^layers\.(\d+)\.", r"blocks.\1."),
        ]
    )
    adapted = transform(weights)
    assert sorted(adapted) == ["blocks.0.proj.proj", "norm"]
    np.testing.assert_array_equal(
        _numpy(adapted["blocks.0.proj.proj"].data),
        np.ones(2, np.float32).view(np.uint8),
    )


def test_ignore_drops_weights_and_logs_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    weights = _checkpoint(
        tmp_path / "model.safetensors",
        {
            "attn.k_scale": np.ones(1, np.float32),
            "attn.v_scale": np.ones(1, np.float32),
            "attn.weight": np.ones(1, np.float32),
        },
    )
    transform = WeightTransform(
        [
            Ignore(r"_scale$", reason="the KV cache does not read them"),
            WeightRenaming(r"^attn\.", "attention."),
        ]
    )
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        adapted = transform(weights)
    assert sorted(adapted) == ["attention.weight"]
    assert "Ignored 2 checkpoint weights matching '_scale$'" in caplog.text
    assert "the KV cache does not read them" in caplog.text


def test_bitcast_relabels_only_weights_stored_as_src(tmp_path: Path) -> None:
    scales = np.arange(6, dtype=np.uint8).reshape(2, 3)
    weights = _checkpoint(
        tmp_path / "model.safetensors",
        {
            "mx.weight_scale": scales,
            "nvfp4.weight_scale": np.ones(3, np.float32),
        },
    )
    transform = WeightTransform(
        [Bitcast(r"\.weight_scale$", DType.uint8, DType.float8_e8m0fnu)]
    )
    adapted = transform(weights)

    mx = adapted["mx.weight_scale"]
    assert isinstance(mx.data, Buffer)
    assert mx.dtype == mx.data.dtype == DType.float8_e8m0fnu
    assert list(mx.shape) == list(mx.data.shape) == [2, 3]
    np.testing.assert_array_equal(_numpy(mx.data), scales)
    assert adapted["nvfp4.weight_scale"].dtype == DType.float32


def test_two_weights_adapting_to_one_name_raise(tmp_path: Path) -> None:
    weights = _checkpoint(
        tmp_path / "model.safetensors",
        {
            "language_model.norm": np.ones(1, np.float32),
            "model.language_model.norm": np.ones(1, np.float32),
        },
    )
    transform = WeightTransform(
        [WeightRenaming(r"^(model\.)?language_model\.", "")]
    )
    with pytest.raises(ValueError, match="both adapt to 'norm'"):
        transform(weights)


@pytest.mark.parametrize(
    ("make_rule", "error"),
    [
        (lambda: Ignore("x", reason=""), "needs a reason"),
        (
            lambda: Bitcast("x", DType.uint8, DType.bfloat16),
            "differ in width",
        ),
        (lambda: WeightRenaming("(", ""), "invalid pattern"),
        (lambda: Ignore(r"^(?!model\.)", reason="r"), "look-around"),
        (lambda: WeightRenaming(r"(a)\1", ""), "back-reference"),
        (lambda: WeightRenaming(r"(\.weight)?$", ""), "empty string"),
    ],
)
def test_malformed_rules_are_refused(
    make_rule: Callable[[], object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        make_rule()


def test_serialize_round_trip() -> None:
    transform = WeightTransform(
        [
            Ignore(r"\.(k|v)_scale$", reason="unused"),
            WeightRenaming(r"^model\.", ""),
            Bitcast(r"\.weight_scale$", DType.uint8, DType.float8_e8m0fnu),
        ]
    )
    serialized = transform.__serialize__()
    assert serialized == {
        "rules": [
            {
                "rule": "Ignore",
                "pattern": r"\.(k|v)_scale$",
                "reason": "unused",
            },
            {
                "rule": "WeightRenaming",
                "pattern": r"^model\.",
                "replacement": "",
            },
            {
                "rule": "Bitcast",
                "pattern": r"\.weight_scale$",
                "src": "uint8",
                "dst": "float8_e8m0fnu",
            },
        ]
    }
    restored = WeightTransform.__deserialize__(
        json.loads(json.dumps(serialized))
    )
    assert restored == transform


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({}, "rules must be a list"),
        ({"rules": [{"rule": "Split", "pattern": "x"}]}, "unknown rule"),
        ({"rules": [{"rule": "Ignore", "pattern": "x"}]}, "takes fields"),
        (
            {
                "rules": [
                    {"rule": "Ignore", "pattern": "x", "reason": "r", "n": "1"}
                ]
            },
            "takes fields",
        ),
        (
            {
                "rules": [
                    {
                        "rule": "Bitcast",
                        "pattern": "x",
                        "src": "u8",
                        "dst": "uint8",
                    }
                ]
            },
            "unknown dtype",
        ),
        (
            {"rules": [{"rule": "Ignore", "pattern": 1, "reason": "r"}]},
            "strings",
        ),
    ],
)
def test_deserialize_refuses_malformed_rules(
    value: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        WeightTransform.__deserialize__(value)


def test_prefix_puts_every_name_under_it(tmp_path: Path) -> None:
    weights = _checkpoint(
        tmp_path / "model.safetensors",
        {
            "layers.0.w": np.ones(1, np.float32),
            "norm": np.ones(1, np.float32),
        },
    )
    rule = WeightRenaming.prefix("target.")
    assert rule == WeightRenaming(r"^(.+)$", r"target.\1")
    assert sorted(WeightTransform([rule])(weights)) == [
        "target.layers.0.w",
        "target.norm",
    ]
    # A prefix is plain text, backslashes included.
    escaped = WeightTransform([WeightRenaming.prefix("a\\1.")])(weights)
    assert "a\\1.norm" in escaped


def test_entries_name_weights_from_their_checkpoints(tmp_path: Path) -> None:
    checkpoints = {
        CheckpointRole.TARGET: _checkpoint(
            tmp_path / "target.safetensors",
            {"model.norm": np.ones(1, np.float32)},
        ),
        CheckpointRole.DRAFT: _checkpoint(
            tmp_path / "draft.safetensors",
            {"model.norm": np.zeros(1, np.float32)},
        ),
    }
    strip = WeightTransform([WeightRenaming(r"^model\.", "")])
    target = WeightEntry(CheckpointRole.TARGET, strip)
    draft = WeightEntry(CheckpointRole.DRAFT, strip)

    weights = apply_weight_entries(
        [target.with_prefix("target."), draft.with_prefix("draft.")],
        checkpoints,
    )
    assert sorted(weights) == ["draft.norm", "target.norm"]
    np.testing.assert_array_equal(
        _numpy(weights["draft.norm"].data),
        np.zeros(1, np.float32).view(np.uint8),
    )
    # Without the prefixes, both checkpoints' norms get one name.
    with pytest.raises(
        ValueError, match=r"two entries name the weights \['norm'\]"
    ):
        apply_weight_entries([target, draft], checkpoints)


def test_weight_naming_round_trip() -> None:
    entry = WeightEntry(
        CheckpointRole.DRAFT, WeightTransform([WeightRenaming(r"^model\.", "")])
    )
    naming = WeightNaming({"spec_decoder": [entry.with_prefix("draft.")]})
    serialized = naming.__serialize__()
    assert serialized == {
        "spec_decoder": [
            {
                "checkpoint": "draft",
                "rules": [
                    {
                        "rule": "WeightRenaming",
                        "pattern": r"^model\.",
                        "replacement": "",
                    },
                    {
                        "rule": "WeightRenaming",
                        "pattern": "^(.+)$",
                        "replacement": r"draft.\1",
                    },
                ],
            }
        ]
    }
    restored = WeightNaming.__deserialize__(json.loads(json.dumps(serialized)))
    assert restored == naming


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({"decoder": {"checkpoint": "target"}}, "list of entries"),
        (
            {"decoder": [{"checkpoint": "base", "rules": []}]},
            "unknown checkpoint",
        ),
        ({"decoder": [{"rules": []}]}, "takes fields"),
    ],
)
def test_weight_naming_refuses_malformed_entries(
    value: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        WeightNaming.__deserialize__(value)
