# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Configuration loader for Qwen2.5VL tests."""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef


class ConfigNames(Enum):
    """Config name."""

    QWEN2_5VL_3B = "qwen2_5vl-3b"


class ConfigLoader:
    """Loader for Qwen2.5VL configs."""

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

    def load_hf_vision_config(self, config_name: ConfigNames) -> dict[str, Any]:
        """Load HuggingFace vision config as dictionary."""
        # Load the full config first
        full_config = self.load_config(config_name)

        # Extract the vision_config section
        if "vision_config" not in full_config:
            raise ValueError(f"Vision config not found in {config_name} config")

        vision_config = full_config["vision_config"].copy()
        # Add missing attributes that tests expect
        vision_config["in_channels"] = vision_config.get("in_chans", 3)

        return vision_config

    def create_qwen2_5vl_config(
        self, config_name: ConfigNames
    ) -> dict[str, Any]:
        """Create Qwen2.5VL config from JSON file."""
        config = self.load_config(config_name)

        # Extract key parameters for MAX implementation
        vision_config = config["vision_config"]

        return {
            "vision_config": {
                "hidden_size": vision_config["hidden_size"],
                "intermediate_size": vision_config["intermediate_size"],
                "num_heads": vision_config["num_heads"],
                "depth": vision_config["depth"],
                "patch_size": vision_config["patch_size"],
                "spatial_merge_size": vision_config["spatial_merge_size"],
                "temporal_patch_size": vision_config["temporal_patch_size"],
                "window_size": vision_config["window_size"],
                "fullatt_block_indexes": vision_config["fullatt_block_indexes"],
                "tokens_per_second": vision_config["tokens_per_second"],
                "in_chans": vision_config.get("in_chans", 3),
                "in_channels": vision_config.get(
                    "in_chans", 3
                ),  # Alternative naming
                "out_hidden_size": vision_config["out_hidden_size"],
                "hidden_act": vision_config["hidden_act"],
                "image_size": 448,  # Default image size for Qwen2.5VL
                "dtype": DType.bfloat16,
            },
            "text_config": {
                "hidden_size": config["hidden_size"],
                "intermediate_size": config["intermediate_size"],
                "num_hidden_layers": config["num_hidden_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "num_key_value_heads": config["num_key_value_heads"],
                "vocab_size": config["vocab_size"],
                "rope_theta": config["rope_theta"],
                "max_position_embeddings": config["max_position_embeddings"],
                "sliding_window": config.get("sliding_window"),
                "rope_scaling": config.get("rope_scaling"),
                "rms_norm_eps": config["rms_norm_eps"],
                "dtype": DType.bfloat16,
            },
            "image_token_id": config["image_token_id"],
            "video_token_id": config["video_token_id"],
            "vision_start_token_id": config["vision_start_token_id"],
            "vision_end_token_id": config["vision_end_token_id"],
            "vision_token_id": config["vision_token_id"],
            "device": DeviceRef.GPU(),
        }

    def get_model_dimensions(self, config_name: ConfigNames) -> dict[str, int]:
        """Get key model dimensions for weight generation."""
        config = self.load_config(config_name)
        vision_config = config["vision_config"]

        return {
            "vision_hidden_size": vision_config["hidden_size"],
            "vision_intermediate_size": vision_config["intermediate_size"],
            "vision_num_layers": vision_config["depth"],
            "vision_num_heads": vision_config["num_heads"],
            "vision_patch_size": vision_config["patch_size"],
            "vision_spatial_merge_size": vision_config["spatial_merge_size"],
            "vision_temporal_patch_size": vision_config["temporal_patch_size"],
            "vision_window_size": vision_config["window_size"],
            "vision_out_hidden_size": vision_config["out_hidden_size"],
            "text_hidden_size": config["hidden_size"],
            "text_intermediate_size": config["intermediate_size"],
            "text_num_layers": config["num_hidden_layers"],
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
