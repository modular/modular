"""Weight adapters for MiniCPMForCausalLM."""
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config=None,
    **kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        # Remap model. → language_model.
        if key.startswith("model."):
            new_key = "language_model." + key[len("model."):]
        elif key == "lm_head.weight":
            new_key = "language_model.lm_head.weight"
        else:
            new_key = key
        new_state_dict[new_key] = value.data()
    return new_state_dict


def convert_gguf_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config=None,
    **kwargs,
) -> dict[str, WeightData]:
    gguf_mapping = {
        "token_embd": "language_model.embed_tokens",
        "blk": "language_model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "language_model.lm_head.weight",
        "output_norm": "language_model.norm",
    }
    new_state_dict: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        new_name = name
        for before, after in gguf_mapping.items():
            new_name = new_name.replace(before, after)
        new_state_dict[new_name] = value.data()
    new_state_dict.pop("rope_freqs.weight", None)
    return new_state_dict