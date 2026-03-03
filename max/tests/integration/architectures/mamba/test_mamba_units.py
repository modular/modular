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
"""Unit tests for Mamba pure-Python helpers (no GPU or model weights)."""

from __future__ import annotations

import pytest


def test_normalize_activation() -> None:
    """_normalize_activation maps activation names correctly."""
    from max.pipelines.architectures.mamba.functional_ops import (
        _normalize_activation,
    )

    assert _normalize_activation("silu") == "silu"
    assert _normalize_activation("SiLU") == "silu"
    assert _normalize_activation("swish") == "silu"
    assert _normalize_activation("Swish") == "silu"
    assert _normalize_activation("none") == "none"
    assert _normalize_activation("") == "none"
    assert _normalize_activation("relu") == "none"


def test_get_state_space_paths_empty_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_state_space_paths returns empty tuple when env var is unset."""
    from max.pipelines.architectures.mamba.functional_ops import (
        _get_state_space_paths,
    )

    # Clear the functools.cache so we get a fresh lookup
    _get_state_space_paths.cache_clear()
    monkeypatch.delenv("MODULAR_MOJO_MAX_IMPORT_PATH", raising=False)
    try:
        result = _get_state_space_paths()
        assert result == ()
    finally:
        _get_state_space_paths.cache_clear()


def test_get_state_space_paths_finds_mojopkg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """_get_state_space_paths finds state_space.mojopkg files."""
    import pathlib

    from max.pipelines.architectures.mamba.functional_ops import (
        _get_state_space_paths,
    )

    # tmp_path is a pathlib.Path from pytest
    p = pathlib.Path(str(tmp_path))
    pkg = p / "state_space.mojopkg"
    pkg.touch()

    _get_state_space_paths.cache_clear()
    monkeypatch.setenv("MODULAR_MOJO_MAX_IMPORT_PATH", str(p))
    try:
        result = _get_state_space_paths()
        assert len(result) == 1
        assert result[0].name == "state_space.mojopkg"
    finally:
        _get_state_space_paths.cache_clear()


def test_mamba_model_inputs_defaults() -> None:
    """MambaModelInputs defaults: is_prefill=True, layer_states=[]."""
    import numpy as np
    from max.driver import Buffer
    from max.pipelines.architectures.mamba.model import (
        MambaModelInputs,
    )

    tokens = Buffer.from_numpy(np.array([1], dtype=np.int64))
    offsets = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32))
    n_logits = Buffer.from_numpy(np.array([1], dtype=np.int64))

    inputs = MambaModelInputs(tokens, offsets, n_logits)
    assert inputs.is_prefill is True
    assert inputs.layer_states == []


def test_mamba_model_inputs_with_states() -> None:
    """MambaModelInputs stores is_prefill=False and layer_states."""
    import numpy as np
    from max.driver import Buffer
    from max.pipelines.architectures.mamba.model import (
        MambaModelInputs,
    )

    tokens = Buffer.from_numpy(np.array([1], dtype=np.int64))
    offsets = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32))
    n_logits = Buffer.from_numpy(np.array([1], dtype=np.int64))
    fake_state = Buffer.from_numpy(np.zeros((1, 4, 3), dtype=np.float32))

    inputs = MambaModelInputs(
        tokens,
        offsets,
        n_logits,
        is_prefill=False,
        layer_states=[fake_state],
    )
    assert inputs.is_prefill is False
    assert len(inputs.layer_states) == 1
