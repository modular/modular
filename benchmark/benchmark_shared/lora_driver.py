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

"""LoRA adapter generation and management utilities for benchmarking."""

import asyncio
import enum
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from safetensors.numpy import save_file
from tqdm.asyncio import tqdm
from transformers import AutoConfig

from .metrics import LoRAMetrics
from .request import async_request_lora_load, async_request_lora_unload

logger = logging.getLogger(__name__)


class LoRAOutputFormat(str, enum.Enum):
    """Output format for LoRA adapter configurations."""

    PATH = "path"  # Returns just paths
    NAME_PATH = "name_path"  # Returns 'name=path' format for TTS benchmarks


@dataclass
class LoRADriver:
    """Configuration for LoRA adapter generation and management.

    Attributes:
        model_id: Base model identifier for generating LoRA adapters.
        lora_rank: LoRA rank (r parameter) for generated adapters.
        num_loras: Number of LoRA adapters to generate (if not using provided paths).
        lora_target_modules: List of module names to apply LoRA to.
        lora_output_dir: Directory to save generated LoRA adapters.
        lora_paths: Optional list of paths to existing LoRA adapters.
        lora_server_path: Optional server path where the server can access LoRA adapters.
            Used when server has different filesystem view (e.g., Docker).
        output_format: Format for returned config values. LoRAOutputFormat.PATH returns just paths,
            LoRAOutputFormat.NAME_PATH returns 'name=path' format for TTS benchmarks.
    """

    model_id: str
    lora_rank: int
    num_loras: int
    lora_target_modules: list[str]
    lora_output_dir: str
    lora_paths: list[str] | None = None
    lora_server_path: str | None = None
    output_format: LoRAOutputFormat = LoRAOutputFormat.PATH

    def _generate_lora_adapter(
        self,
        output_dir: str,
        adapter_name: str | None = None,
    ) -> None:
        """Generate a minimal LoRA adapter for testing.

        Args:
            output_dir: Directory to save the adapter files.
            adapter_name: Optional name for the adapter (used in metadata).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load base model config to get dimensions
        config = AutoConfig.from_pretrained(self.model_id)

        # Create adapter config
        # TODO: Make this a dataclass type than a dict[str, Any]. It's really
        # hard to typecheck things if we missed a param somewhere.
        adapter_config: dict[str, Any] = {
            "alpha_pattern": {},
            "auto_mapping": None,
            "base_model_name_or_path": self.model_id,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": self.lora_rank,
            "rank_pattern": {},
            "revision": None,
            "target_modules": self.lora_target_modules,
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }

        # Save adapter config
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Generate minimal LoRA weights
        lora_weights = {}

        # For each layer and target module, create LoRA A and B matrices
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        # Handle grouped query attention - k_proj and v_proj have different dimensions
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads

        # Determine output dimensions for attention projection modules
        attn_module_dims = {
            "q_proj": hidden_size,  # hidden_size -> hidden_size
            "k_proj": num_kv_heads * head_dim,  # hidden_size -> kv_hidden_size
            "v_proj": num_kv_heads * head_dim,  # hidden_size -> kv_hidden_size
            "o_proj": hidden_size,  # hidden_size -> hidden_size
        }

        for layer_idx in range(num_layers):
            for module in self.lora_target_modules:
                # Validate that module is supported (attention only for now)
                if module not in attn_module_dims:
                    raise ValueError(
                        f"Unsupported target module '{module}'. Only attention"
                        " modules are currently supported:"
                        f" {list(attn_module_dims.keys())}"
                    )

                # Get the output dimension for this module type
                out_dim = attn_module_dims[module]
                in_dim = hidden_size  # Input is always hidden_size for attention modules

                # LoRA A: shape should be (lora_rank, in_dim)
                lora_a_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
                lora_weights[lora_a_key] = (
                    np.random.randn(self.lora_rank, in_dim).astype(np.float32)
                    * 0.01
                )

                # LoRA B: shape should be (out_dim, lora_rank)
                # Use zeros for compatibility with benchmark_serving.py behavior
                # (benchmark_tts_serving.py uses random, but zeros is safer for testing)
                lora_b_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
                lora_weights[lora_b_key] = np.zeros(
                    (out_dim, self.lora_rank), dtype=np.float32
                )

        # Save weights in safetensors format
        save_file(
            lora_weights, os.path.join(output_dir, "adapter_model.safetensors")
        )

    def generate_loras(self) -> dict[str, str]:
        """Generate LoRA adapter configurations.

        Returns:
            Dictionary mapping adapter names to paths or 'name=path' strings,
            depending on output_format setting.
        """
        lora_configs: dict[str, str] = {}

        if self.lora_paths:
            # Use provided LoRA paths
            logger.info("Using provided LoRA paths")
            for i, path in enumerate(self.lora_paths):
                # Support both "name=path" and just "path" formats
                if "=" in path:
                    name, path = path.split("=", 1)
                else:
                    name = f"adapter_{i}"

                abs_path = os.path.abspath(path)

                # Format output based on output_format setting
                if self.output_format == LoRAOutputFormat.NAME_PATH:
                    lora_configs[name] = f"{name}={abs_path}"
                else:
                    lora_configs[name] = abs_path
        else:
            # Generate test LoRA adapters
            logger.info(f"Preparing {self.num_loras} test LoRA adapters...")

            # Use custom output directory if specified, otherwise use temp directory
            base_output_dir = os.path.abspath(
                os.path.expanduser(self.lora_output_dir)
            )
            os.makedirs(base_output_dir, exist_ok=True)

            for i in range(self.num_loras):
                adapter_name = f"generated_adapter_{i}"
                adapter_path = os.path.join(base_output_dir, adapter_name)

                self._generate_lora_adapter(
                    output_dir=adapter_path,
                    adapter_name=adapter_name,
                )

                # Use server path if specified, otherwise use absolute local path
                if self.lora_server_path:
                    relative_path = os.path.relpath(
                        adapter_path, base_output_dir
                    )
                    server_path = os.path.join(
                        self.lora_server_path, relative_path
                    )
                else:
                    server_path = os.path.abspath(adapter_path)

                # Format output based on output_format setting
                if self.output_format == LoRAOutputFormat.NAME_PATH:
                    lora_configs[adapter_name] = f"{adapter_name}={server_path}"
                else:
                    lora_configs[adapter_name] = server_path

        return lora_configs


async def benchmark_lora_loading(
    api_url: str,
    lora_configs: dict[str, str],
    metrics: LoRAMetrics,
    max_concurrent: int = 1,
) -> None:
    """Benchmark LoRA loading performance.

    Args:
        api_url: Base API URL
        lora_configs: Dictionary mapping adapter names to paths
        metrics: LoRAMetrics instance to record loading performance
        max_concurrent: Maximum concurrent loading operations
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def load_with_semaphore(name: str, path: str) -> None:
        async with semaphore:
            success, load_time = await async_request_lora_load(
                api_url, name, path
            )
            if success:
                metrics.load_times_ms.append(load_time)
                metrics.total_loads += 1
            else:
                logger.warning(f"Failed to load LoRA '{name}'")

    tasks = [
        load_with_semaphore(name, path) for name, path in lora_configs.items()
    ]
    await tqdm.gather(*tasks, desc="Loading LoRAs...")


async def benchmark_lora_unloading(
    api_url: str,
    lora_configs: dict[str, str],
    metrics: LoRAMetrics,
    max_concurrent: int = 1,
) -> None:
    """Benchmark LoRA unloading performance.

    Args:
        api_url: Base API URL
        lora_configs: Dictionary mapping adapter names to paths (names are used)
        metrics: LoRAMetrics instance to record unloading performance
        max_concurrent: Maximum concurrent unloading operations
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def unload_with_semaphore(name: str) -> None:
        async with semaphore:
            success, unload_time = await async_request_lora_unload(
                api_url, name
            )
            if success:
                metrics.unload_times_ms.append(unload_time)
                metrics.total_unloads += 1
            else:
                logger.warning(f"Failed to unload LoRA '{name}'")

    tasks = [unload_with_semaphore(name) for name in lora_configs]
    await tqdm.gather(*tasks, desc="Unloading LoRAs...")
