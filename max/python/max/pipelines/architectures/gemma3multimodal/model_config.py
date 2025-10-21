# Handles model configuration and parameter parsing

# do we need to mimic what internvl does in:
# `_select_llm_config_class()`
# not sure since that refers to huggingface config and our impl isn't on there yet

# the model config class should `generate()` both the language and vision configs

# for starters, just trying to use the standard Gemma3 one

from dataclasses import dataclass
from max.pipelines.lib import MAXModelConfig
from ..gemma3 import Gemma3ConfigBase, Gemma3TextConfig

@dataclass
class Gemma3VisionConfig:
    """
    The vision-specific config for Gemma3
    fields and defaults taken from below link - unsure if they are valid
    https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json
    """
    hidden_size: int = 1152
    """Dimensionality of the encoder layers and the pooler layer"""
    image_size: int = 896
    """The size (resolution) of each image"""
    intermediate_size: int = 4304
    """Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder"""
    model_type: str = "siglip_vision_model"
    """maybe not required?"""
    num_attention_heads: int = 16
    """Number of attention heads for each attention layer in the Transformer encoder"""
    num_hidden_layers: int = 27
    """Number of hidden layers in the Transformer encoder"""
    patch_size: int = 14
    """The size (resolution) of each patch"""
    vision_use_head: bool = False
    """maybe not required?"""

    @staticmethod
    def generate(
        vision_config: AutoConfig
    ) -> Gemma3VisionConfig:
        hidden_size = vision_config.hidden_size
        return Gemma3VisionConfig(
            hidden_size=vision_config.hidden_size,
            image_size=vision_config.image_size,
            intermediate_size=vision_config.intermediate_size,
            num_attention_heads=vision_config.num_attention_heads,
            num_hidden_layers=getattr(vision_config, "num_hidden_layers", 32),
            patch_size=vision_config.patch_size,
        )

# TODO don't know what's goin on with this... look into what MAXModelConfig needs
# to add on top of Gemma3VisionConfig?
@dataclass
class Gemma3VLConfig(MAXModelConfig):
    # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/configuration_gemma3.py
    vision_config: Gemma3VisionConfig # siglip vision model stuff?
    """Custom vision config or dict"""
    text_config: Gemma3TextConfig = None
    """The config object of the text backbone"""
    mm_tokens_per_image: int = 256
    """The number of tokens per image embedding"""
    boi_token_index: int = 255_999
    """The begin-of-image token index to wrap the image prompt"""
    eoi_token_index: int = 256_000
    """The end-of-image token index to wrap the image prompt"""
    image_token_index: int = 262_144
    """The image token index to encode the image prompt"""
    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer for initializing all weight matrices"""

    def help() -> str:
        return "Gemma3 Vision-Language model configuration."

    def generate(
        vision_config: Gemma3VisionConfig,
        text_config: Gemma3TextConfig,
        mm_tokens_per_image: int,
        boi_token_index: int,
        eoi_token_index: int,
        image_token_index: int,
        intializer_range: float,
    ) -> Gemma3VLConfig:
        return Gemma3VLConfig(
            vision_config=vision_config,
            text_config=text_config,
            mm_tokens_per_image=mm_tokens_per_image,
            boi_token_index=boi_token_index,
            eoi_token_index=eoi_token_index,
            image_token_index=image_token_index,
            intializer_range=initializer_range,
        )