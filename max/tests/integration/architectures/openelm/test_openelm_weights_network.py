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
"""Verify the real checkpoint's weight keys match model.py's expectations.

Downloads apple/OpenELM-270M-Instruct via huggingface_hub on first run
(cached afterward). Kept in its own file so BUILD.bazel can glob it into
a separate target tagged "requires-network".
"""


class TestModelWeightFiles:
    REPO_ID = "apple/OpenELM-270M-Instruct"

    def _model_path(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(self.REPO_ID)

    def test_weight_file_exists(self) -> None:
        from pathlib import Path

        model_path = Path(self._model_path())
        weight_file = model_path / "model.safetensors"
        assert weight_file.exists(), (
            f"model.safetensors not found at {weight_file}"
        )

    def test_embedding_weight_exists(self) -> None:
        from max.pipelines.architectures.openelm.weight_adapters import (
            list_weight_names,
        )

        names = set(list_weight_names(self._model_path()))
        assert "transformer.token_embeddings.weight" in names

    def test_final_norm_weight_exists(self) -> None:
        from max.pipelines.architectures.openelm.weight_adapters import (
            list_weight_names,
        )

        names = set(list_weight_names(self._model_path()))
        assert "transformer.norm.weight" in names

    def test_layer_zero_weights_exist(self) -> None:
        from max.pipelines.architectures.openelm.weight_adapters import (
            list_weight_names,
        )

        names = set(list_weight_names(self._model_path()))

        required = [
            "transformer.layers.0.attn_norm.weight",
            "transformer.layers.0.attn.qkv_proj.weight",
            "transformer.layers.0.attn.out_proj.weight",
            "transformer.layers.0.ffn_norm.weight",
            "transformer.layers.0.ffn.proj_1.weight",
            "transformer.layers.0.ffn.proj_2.weight",
        ]

        for key in required:
            assert key in names, (
                f"Required weight '{key}' not found in model.safetensors"
            )

    def test_config_json_exists_and_readable(self) -> None:
        import json
        from pathlib import Path

        model_path = Path(self._model_path())
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        with open(config_file) as f:
            config = json.load(f)

        assert "architectures" in config
        assert "OpenELMForCausalLM" in config["architectures"], (
            f"Expected 'OpenELMForCausalLM' in architectures, "
            f"got: {config['architectures']}"
        )
