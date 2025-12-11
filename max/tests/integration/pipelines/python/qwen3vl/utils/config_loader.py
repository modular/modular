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

"""Configuration loader for Qwen3VL-30B-A3B-Instruct tests.

The config file is copied from the original config file:
https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/config.json
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef


class ConfigNames(Enum):
    """Config name."""

    QWEN3VL_30B = "qwen3-VL-30B-A3B-Instruct"


class ConfigLoader:
    """Loader for Qwen3VL configs."""

    def __init__(self, config_dir: Path):
        """Initialize with config directory path."""
        self.config_dir = config_dir
        self._cache: dict[str, dict[str, Any]] = {}

    def load_config(self, config_name: ConfigNames) -> dict[str, Any]:
        """Load config from JSON file."""
        if config_name.value not in self._cache:
            config_path = self.config_dir / f"{config_name.value}.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                self._cache[config_name.value] = json.load(f)

        return self._cache[config_name.value]

    def create_qwen3vl_config(self, config_name: ConfigNames) -> dict[str, Any]:
        """Create Qwen3VL-30B-A3B-Instruct config from JSON file."""
        config = self.load_config(config_name)

        # Extract key parameters for MAX implementation
        vision_config = config["vision_config"]
        text_config = config["text_config"]
        return {
            "vision_config": {
                "hidden_size": vision_config["hidden_size"],
                "intermediate_size": vision_config["intermediate_size"],
                "num_heads": vision_config["num_heads"],
                "depth": vision_config["depth"],
                "patch_size": vision_config["patch_size"],
                "spatial_merge_size": vision_config["spatial_merge_size"],
                "temporal_patch_size": vision_config["temporal_patch_size"],
                "num_position_embeddings": vision_config[
                    "num_position_embeddings"
                ],
                "deepstack_visual_indexes": vision_config[
                    "deepstack_visual_indexes"
                ],
                "in_channels": vision_config.get(
                    "in_channels", 3
                ),  # Alternative naming
                "out_hidden_size": vision_config["out_hidden_size"],
                "hidden_act": vision_config["hidden_act"],
                "dtype": DType.bfloat16,
            },
            "text_config": {
                "hidden_size": text_config["hidden_size"],
                "intermediate_size": text_config["intermediate_size"],
                "num_hidden_layers": text_config["num_hidden_layers"],
                "num_attention_heads": text_config["num_attention_heads"],
                "num_key_value_heads": text_config["num_key_value_heads"],
                "vocab_size": text_config["vocab_size"],
                "rope_theta": text_config["rope_theta"],
                "max_position_embeddings": text_config[
                    "max_position_embeddings"
                ],
                "rope_scaling": text_config.get("rope_scaling"),
                "rms_norm_eps": text_config["rms_norm_eps"],
                "dtype": DType.bfloat16,
                "head_dim": text_config["head_dim"],
            },
            "image_token_id": config["image_token_id"],
            "video_token_id": config["video_token_id"],
            "vision_start_token_id": config["vision_start_token_id"],
            "vision_end_token_id": config["vision_end_token_id"],
            "device": DeviceRef.GPU(),
        }


# Global instance
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """Get global config loader instance."""
    global _config_loader
    if _config_loader is None:
        config_dir = Path(__file__).parent.parent / "configs"
        _config_loader = ConfigLoader(config_dir)
    return _config_loader
