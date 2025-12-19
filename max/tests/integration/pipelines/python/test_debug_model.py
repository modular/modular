# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

import max.tests.integration.tools.debug_model as dbg
import pytest
import transformers


def test_apply_config_overrides() -> None:
    """Test that _apply_config_overrides works for config objects."""
    config = transformers.AutoConfig.for_model("gpt2")
    original_n_head = config.n_head

    dbg._apply_config_overrides(
        config, {"n_embd": 2048, "n_layer": 24}, "config"
    )

    assert config.n_embd == 2048
    assert config.n_layer == 24
    assert config.n_head == original_n_head  # unchanged

    config2 = transformers.AutoConfig.for_model("gpt2")
    with pytest.raises(ValueError, match=r"Invalid override key"):
        dbg._apply_config_overrides(config2, {"invalid_key": 999}, "config")


def test_apply_hf_config_override() -> None:
    """Test apply_hf_config_override context manager."""
    base_cfg = transformers.AutoConfig.for_model("gpt2")
    orig_prop = dbg.MAXModelConfig.huggingface_config

    real_cfg = dbg.MAXModelConfig(
        model_path="dummy/text",
        _huggingface_config=base_cfg,
    )

    original_n_layer = base_cfg.n_layer

    with dbg.apply_hf_config_override({"n_embd": 2048}):
        assert dbg.MAXModelConfig.huggingface_config is not orig_prop
        cfg_in_ctx = real_cfg.huggingface_config
        assert cfg_in_ctx.n_embd == 2048
        assert cfg_in_ctx.n_layer == original_n_layer

    assert dbg.MAXModelConfig.huggingface_config is orig_prop

    fresh_cfg = transformers.AutoConfig.for_model("gpt2")
    real_cfg2 = dbg.MAXModelConfig(
        model_path="dummy/text",
        _huggingface_config=fresh_cfg,
    )
    cfg_after = real_cfg2.huggingface_config
    assert cfg_after.n_embd == fresh_cfg.n_embd

    real_cfg3 = dbg.MAXModelConfig(
        model_path="dummy/text",
        _huggingface_config=transformers.AutoConfig.for_model("gpt2"),
    )
    with pytest.raises(ValueError, match=r"Invalid override key"):
        with dbg.apply_hf_config_override({"not_a_key": 1}):
            _ = real_cfg3.huggingface_config


def test_apply_non_strict_load() -> None:
    """Test apply_non_strict_load context manager."""
    orig_module_load = dbg.Module.load_state_dict
    orig_module_v3_load = getattr(dbg.ModuleV3, "load_state_dict", None)

    strict_values: list[bool | None] = []

    def mock_load_state_dict(self: Any, *args: Any, **kwargs: Any) -> Any:
        strict_values.append(kwargs.get("strict"))
        return None

    with patch.object(dbg.Module, "load_state_dict", mock_load_state_dict):
        with dbg.apply_non_strict_load():
            module_mock = Mock(spec=dbg.Module)
            dbg.Module.load_state_dict(module_mock, {})
            assert strict_values[-1] is False

            dbg.Module.load_state_dict(module_mock, {}, strict=True)
            assert strict_values[-1] is False

        assert dbg.Module.load_state_dict == mock_load_state_dict

    assert dbg.Module.load_state_dict == orig_module_load
    if orig_module_v3_load is not None:
        assert dbg.ModuleV3.load_state_dict == orig_module_v3_load


def test_apply_max_hooks_without_output_dir() -> None:
    """Test apply_max_hooks creates and cleans up hook without output directory."""
    orig_infer_init = dbg.InferenceSession.__init__
    with dbg.apply_max_hooks(output_directory=None) as hook:
        assert isinstance(hook, dbg.PrintHook)
        assert dbg.InferenceSession.__init__ == orig_infer_init
    assert dbg.InferenceSession.__init__ == orig_infer_init


def test_apply_max_hooks_with_output_dir() -> None:
    """Test apply_max_hooks patches InferenceSession when output_directory is provided."""
    orig_infer_init = dbg.InferenceSession.__init__
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        with dbg.apply_max_hooks(output_directory=output_path) as hook:
            assert isinstance(hook, dbg.PrintHook)
            assert dbg.InferenceSession.__init__ != orig_infer_init
        assert dbg.InferenceSession.__init__ == orig_infer_init


def test_debug_context_with_hf_overrides() -> None:
    """Test debug_context with HF config overrides."""
    base_cfg = transformers.AutoConfig.for_model("gpt2")
    orig_prop = dbg.MAXModelConfig.huggingface_config
    orig_module_load = dbg.Module.load_state_dict

    real_cfg = dbg.MAXModelConfig(
        model_path="dummy/text",
        _huggingface_config=base_cfg,
    )

    with dbg.debug_context(
        output_directory=None,
        hf_config_overrides={"n_embd": 3072},
    ):
        assert dbg.MAXModelConfig.huggingface_config is not orig_prop
        cfg_in_ctx = real_cfg.huggingface_config
        assert cfg_in_ctx.n_embd == 3072

        assert dbg.Module.load_state_dict != orig_module_load

    assert dbg.MAXModelConfig.huggingface_config is orig_prop
    assert dbg.Module.load_state_dict == orig_module_load


def test_debug_context_without_hf_overrides() -> None:
    """Test debug_context without HF config overrides."""
    base_cfg = transformers.AutoConfig.for_model("gpt2")
    orig_prop = dbg.MAXModelConfig.huggingface_config
    orig_module_load = dbg.Module.load_state_dict

    real_cfg = dbg.MAXModelConfig(
        model_path="dummy/text",
        _huggingface_config=base_cfg,
    )
    with dbg.debug_context(
        output_directory=None,
        hf_config_overrides=None,
    ):
        assert dbg.MAXModelConfig.huggingface_config is orig_prop
        cfg_in_ctx = real_cfg.huggingface_config
        assert cfg_in_ctx.n_embd == base_cfg.n_embd  # unchanged
        assert dbg.Module.load_state_dict != orig_module_load
    assert dbg.MAXModelConfig.huggingface_config is orig_prop
    assert dbg.Module.load_state_dict == orig_module_load


def test_debug_context_with_output_directory(tmp_path: Path) -> None:
    """Test debug_context with output directory patches InferenceSession."""
    orig_infer_init = dbg.InferenceSession.__init__

    with dbg.debug_context(
        output_directory=tmp_path,
        hf_config_overrides=None,
    ):
        assert dbg.InferenceSession.__init__ != orig_infer_init
    assert dbg.InferenceSession.__init__ == orig_infer_init
