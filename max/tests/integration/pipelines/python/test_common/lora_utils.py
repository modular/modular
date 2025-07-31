# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for LoRA testing across pipeline tests."""

import json
import os
import tempfile
from typing import Any, Optional

import hf_repo_lock
import torch
from max.driver import DeviceSpec
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextGenerationPipeline,
    TextTokenizer,
)
from safetensors.torch import save_file
from transformers import AutoConfig

# Common constants
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)


def generate_test_lora_adapter(
    base_model_id: str,
    output_dir: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[list[str]] = None,
) -> None:
    """Generate a minimal LoRA adapter for testing.

    Args:
        base_model_id: HuggingFace model ID to generate adapter for
        output_dir: Directory to save the adapter files
        lora_rank: LoRA rank (r parameter)
        lora_alpha: LoRA alpha parameter for scaling
        target_modules: List of module names to apply LoRA to
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load base model config to get dimensions
    config = AutoConfig.from_pretrained(base_model_id)

    # Default target modules for transformer models
    # Use only q_proj to avoid GQA complexity with k_proj/v_proj
    if target_modules is None:
        target_modules = ["q_proj"]

    # Create adapter config
    adapter_config: dict[str, Any] = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
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
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(
        config, "num_key_value_heads", num_attention_heads
    )
    head_dim = hidden_size // num_attention_heads

    for layer_idx in range(num_layers):
        for module in target_modules:
            # Use full hidden_size for all modules to match LoRA manager expectations
            proj_dim = hidden_size

            # LoRA A: maps from proj_dim to rank
            lora_a_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            # Initialize with small random values
            lora_weights[lora_a_key] = torch.randn(lora_rank, proj_dim) * 0.01

            # LoRA B: maps from rank to proj_dim
            lora_b_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            # Initialize to zero (common practice)
            lora_weights[lora_b_key] = torch.zeros(proj_dim, lora_rank)

    # Save weights in safetensors format
    save_file(
        lora_weights, os.path.join(output_dir, "adapter_model.safetensors")
    )


def create_test_lora_adapter(
    base_model_id: str = REPO_ID,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[list[str]] = None,
    prefix: str = "lora_test",
) -> str:
    """Create a temporary LoRA adapter for testing.

    Returns:
        str: Path to the generated LoRA adapter directory
    """
    temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
    generate_test_lora_adapter(
        base_model_id=base_model_id,
        output_dir=temp_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )
    return temp_dir


def create_multiple_test_lora_adapters(
    base_model_id: str = REPO_ID,
    num_adapters: int = 2,
    prefix: str = "lora_test_multi",
) -> list[str]:
    """Create multiple temporary LoRA adapters for testing.

    Returns:
        List[str]: List of paths to the generated LoRA adapter directories
    """
    adapters = []

    for i in range(num_adapters):
        # Vary the configuration slightly for each adapter
        lora_rank = 8 if i == 0 else 4
        lora_alpha = 16 if i == 0 else 8

        temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_{i}_")
        generate_test_lora_adapter(
            base_model_id=base_model_id,
            output_dir=temp_dir,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj"],
        )
        adapters.append(temp_dir)

    return adapters


def create_pipeline_config_with_lora(
    lora_paths: list[str],
    model_path: str = REPO_ID,
    max_num_loras: int = 2,
    max_lora_rank: int = 16,
) -> PipelineConfig:
    """Create a pipeline configuration with LoRA enabled.

    Args:
        lora_paths: List of paths to LoRA adapters
        model_path: Path to the base model
        max_num_loras: Maximum number of LoRA adapters
        max_lora_rank: Maximum LoRA rank

    Returns:
        PipelineConfig: Configuration with LoRA settings
    """
    return PipelineConfig(
        model_path=model_path,
        quantization_encoding=SupportedEncoding.bfloat16,  # Use bfloat16 for GPU
        max_batch_size=4,
        max_num_loras=max_num_loras,
        lora_paths=lora_paths,
        max_lora_rank=max_lora_rank,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        max_length=512,
        cache_strategy=KVCacheStrategy.PAGED,
        device_specs=[DeviceSpec(device_type="gpu", id=0)],
    )


def create_pipeline_config_base(model_path: str = REPO_ID) -> PipelineConfig:
    """Create a base pipeline configuration without LoRA.

    Args:
        model_path: Path to the base model

    Returns:
        PipelineConfig: Base configuration without LoRA
    """
    return PipelineConfig(
        model_path=model_path,
        quantization_encoding=SupportedEncoding.bfloat16,  # Use bfloat16 for GPU
        max_batch_size=4,
        max_length=512,
        cache_strategy=KVCacheStrategy.PAGED,
        allow_safetensors_weights_fp32_bf6_bidirectional_cast=True,
        device_specs=[DeviceSpec(device_type="gpu", id=0)],
    )


def create_pipeline_with_lora(lora_paths: list[str]) -> TextGenerationPipeline:
    """Create a text generation pipeline with LoRA enabled.

    Args:
        lora_paths: List of paths to LoRA adapters

    Returns:
        TextGenerationPipeline: Pipeline with LoRA adapters loaded
    """
    config = create_pipeline_config_with_lora(lora_paths)
    _, pipeline = PIPELINE_REGISTRY.retrieve(config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


def create_pipeline_base() -> TextGenerationPipeline:
    """Create a base text generation pipeline without LoRA.

    Returns:
        TextGenerationPipeline: Base pipeline without LoRA
    """
    config = create_pipeline_config_base()
    _, pipeline = PIPELINE_REGISTRY.retrieve(config)
    assert isinstance(pipeline, TextGenerationPipeline)
    return pipeline


def create_tokenizer(
    model_path: str = REPO_ID, max_length: int = 512
) -> TextTokenizer:
    """Create a tokenizer for the specified model.

    Args:
        model_path: Path to the model
        max_length: Maximum sequence length

    Returns:
        TextTokenizer: Tokenizer instance
    """
    return TextTokenizer(
        model_path,
        revision=REVISION,
        max_length=max_length,
        max_new_tokens=max_length,
        trust_remote_code=True,
    )


def get_model_revision() -> str:
    """Get the locked model revision for testing.

    Returns:
        str: Model revision from hf-repo-lock.tsv
    """
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    return REVISION
