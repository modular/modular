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

from __future__ import annotations

from collections.abc import Callable

import max.driver as md
import pytest
import torch
from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.umt5.model_config import UMT5ConfigBase
from max.pipelines.architectures.umt5.umt5 import UMT5EncoderModel
from max.tensor import Tensor
from torch.utils.dlpack import from_dlpack
from transformers import UMT5Config
from transformers.models.umt5.modeling_umt5 import UMT5EncoderModel as HfUMT5


def _tiny_umt5_config_dict() -> dict[str, int | float | str]:
    return {
        "vocab_size": 32128,
        "d_model": 64,
        "d_kv": 16,
        "d_ff": 128,
        "num_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 4,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "dropout_rate": 0.0,
        "layer_norm_epsilon": 1e-6,
        "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu",
    }


def _adapt_hf_weights_for_max(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    weights = dict(hf_state_dict)
    shared = weights.get("shared.weight")
    embed = weights.get("encoder.embed_tokens.weight")

    # MAX registers both keys, while some HF checkpoints may only carry one.
    if shared is None and embed is not None:
        weights["shared.weight"] = embed
    if embed is None and shared is not None:
        weights["encoder.embed_tokens.weight"] = shared
    return weights


def _diffusers_like_get_t5_prompt_embeds_from_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    num_videos_per_prompt: int,
    max_sequence_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror diffusers WanPipeline._get_t5_prompt_embeds postprocessing."""
    prompt_embeds = hidden_states.to(dtype=dtype, device=hidden_states.device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = [
        u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)
    ]
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [
                    u,
                    u.new_zeros(
                        max_sequence_length - u.size(0),
                        u.size(1),
                    ),
                ]
            )
            for u in prompt_embeds
        ],
        dim=0,
    )

    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_videos_per_prompt, seq_len, -1
    )
    return prompt_embeds


def _diffusers_like_get_t5_prompt_embeds_from_encoder(
    encoder_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_videos_per_prompt: int,
    max_sequence_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror WanPipeline._get_t5_prompt_embeds including encoder call."""
    hidden_states = encoder_fn(input_ids, attention_mask)
    return _diffusers_like_get_t5_prompt_embeds_from_hidden(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        dtype=dtype,
    )


@torch.no_grad()
def test_umt5_encoder_matches_transformers() -> None:
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")

    torch.manual_seed(42)
    cfg = _tiny_umt5_config_dict()

    hf_model = HfUMT5(UMT5Config(**cfg)).to(torch.bfloat16).to("cuda").eval()
    hf_state = {
        key: value.detach().to(torch.bfloat16).to("cuda")
        for key, value in hf_model.state_dict().items()
    }
    max_weights = _adapt_hf_weights_for_max(hf_state)

    max_config = UMT5ConfigBase(
        **cfg,
        dtype=DType.bfloat16,
        device=DeviceRef.GPU(),
    )

    with F.lazy():
        max_model = UMT5EncoderModel(max_config)
        max_model.to(Accelerator())

    required_weight_keys = {name for name, _ in max_model.parameters}
    missing = sorted(required_weight_keys - set(max_weights.keys()))
    assert not missing, f"Missing weights for MAX UMT5: {missing[:8]}"

    max_model.load_state_dict(max_weights, strict=False)
    compiled = max_model.compile(*max_model.input_types())

    input_ids = torch.randint(
        0,
        int(cfg["vocab_size"]),
        (2, 10),
        dtype=torch.int64,
        device="cuda",
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    hf_output = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    max_output = compiled(
        Tensor.from_dlpack(input_ids),
        Tensor.from_dlpack(attention_mask),
    )
    max_output_torch = from_dlpack(max_output).to(torch.bfloat16)

    assert bool(torch.isfinite(hf_output).all()), "HF output contains NaN/Inf."
    assert bool(torch.isfinite(max_output_torch).all()), (
        "MAX output contains NaN/Inf."
    )

    abs_diff = (
        hf_output.to(torch.float32) - max_output_torch.to(torch.float32)
    ).abs()
    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())

    assert max_abs_diff < 0.2, f"max_abs_diff too large: {max_abs_diff:.6f}"
    assert mean_abs_diff < 0.04, f"mean_abs_diff too large: {mean_abs_diff:.6f}"

    # WanPipeline._get_t5_prompt_embeds-like parity check:
    # encoder(input_ids, attention_mask) + trim/pad/repeat.
    num_videos_per_prompt = 3
    max_sequence_length = int(input_ids.shape[1])
    hf_prompt_embeds = _diffusers_like_get_t5_prompt_embeds_from_encoder(
        encoder_fn=lambda ids, mask: hf_model(
            input_ids=ids, attention_mask=mask
        ).last_hidden_state,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        dtype=torch.bfloat16,
    )
    max_prompt_embeds = _diffusers_like_get_t5_prompt_embeds_from_encoder(
        encoder_fn=lambda ids, mask: from_dlpack(
            compiled(
                Tensor.from_dlpack(ids),
                Tensor.from_dlpack(mask),
            )
        ).to(torch.bfloat16),
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        dtype=torch.bfloat16,
    )

    prompt_abs_diff = (
        hf_prompt_embeds.to(torch.float32) - max_prompt_embeds.to(torch.float32)
    ).abs()
    prompt_max_abs_diff = float(prompt_abs_diff.max().item())
    prompt_mean_abs_diff = float(prompt_abs_diff.mean().item())

    assert prompt_max_abs_diff < 0.2, (
        f"diffusers_like_prompt max_abs_diff too large: {prompt_max_abs_diff:.6f}"
    )
    assert prompt_mean_abs_diff < 0.04, (
        f"diffusers_like_prompt mean_abs_diff too large: {prompt_mean_abs_diff:.6f}"
    )
