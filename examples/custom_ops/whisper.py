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

# run with: `br //PyTorch:repl -- $(realpath whisper.py)`

import os
import sys
import sysconfig
from pathlib import Path
from typing import Optional

import max.torch_legacy as mtorch  # type: ignore
import torch
import transformers
from datasets import load_dataset
from max import engine
from max.driver import Accelerator
from torch import nn
from transformers import (
    CompileConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.modeling_whisper import (
    EncoderDecoderCache,
    WhisperConfig,
    WhisperEncoderLayer,
)

# Setup python for nested mojo runs
os.environ["MOJO_PYTHON"] = sys.executable
os.environ["MOJO_PYTHON_LIBRARY"] = (
    Path(sys.executable).resolve().parent.parent
    / "lib"
    / sysconfig.get_config_var("INSTSONAME")
).as_posix()


class ModularWhisperAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[WhisperConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]
    ]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        Q = (
            self.q_proj(hidden_states)
            .mul(self.scaling)  # Apply scaling factor
            .view(tgt_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        K = (
            self.k_proj(hidden_states)
            .view(tgt_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        V = (
            self.v_proj(hidden_states)
            .view(tgt_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )

        # Call custom fused attention op
        results = []
        for head_idx in range(self.num_heads):
            results.append(
                torch.ops.modular_ops.fused_attention_custom(
                    Q[head_idx, :, :],
                    K[head_idx, :, :],
                    V[head_idx, :, :],
                    mojo_parameters={
                        "BN": 16,
                        "BD": 8,
                    },
                )
            )
        O = torch.stack(results, dim=0)
        O = O.transpose(0, 1)
        O = O.reshape(bsz, tgt_len, self.embed_dim)
        O = self.out_proj(O)
        return O, None, None


def get_model(device, backend):
    # Load model configuration first.
    config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")
    # Modify to get more GPU-friendly tensor shapes (here, divisible by 8).
    # This allows our Mojo attention kernel to not require border handling.
    # Default is 1500.
    config.max_source_positions = 1504
    # Load the pretrained model.
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en",
        attn_implementation="eager",
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Replace all WhisperAttention layers with ModularWhisperAttention
    for name, module in model.named_modules():
        if isinstance(
            module,
            transformers.models.whisper.modeling_whisper.WhisperAttention,
        ):
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)

            # Only replace attention in encoder layers, since these match
            # the shape constraints of the custom op
            if not isinstance(
                parent,
                WhisperEncoderLayer,
            ):
                continue

            # Create ModularWhisperAttention with same config
            new_attention = ModularWhisperAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                is_decoder=module.is_decoder,
                bias=True,
                is_causal=module.is_causal,
                layer_idx=module.layer_idx,
                config=module.config,
            )

            # Copy weights from old attention
            new_attention.k_proj.weight.data = module.k_proj.weight.data

            new_attention.v_proj.weight.data = module.v_proj.weight.data
            new_attention.v_proj.bias.data = module.v_proj.bias.data

            new_attention.q_proj.weight.data = module.q_proj.weight.data
            new_attention.q_proj.bias.data = module.q_proj.bias.data

            new_attention.out_proj.weight.data = module.out_proj.weight.data
            new_attention.out_proj.bias.data = module.out_proj.bias.data

            # Replace the attention module
            setattr(parent, layer_name, new_attention)

    model.to(device).eval()

    # Enable static cache and compile the forward pass
    model.generation_config.cache_implementation = "static"
    model.generation_config.compile_config = CompileConfig(backend=backend)

    return model


def main():
    if not torch.cuda.is_available():
        print("This example is only available for GPUs at the moment.")
        return

    # Get the path to our Mojo custom ops
    mojo_kernels = Path(__file__).parent / "kernels"

    inference_session = engine.InferenceSession(
        devices=[Accelerator()],
        custom_extensions=[mojo_kernels],
    )
    with torch.no_grad():
        mtorch.register_custom_ops(inference_session)

    device = torch.device("cuda:0")

    model = get_model(device, "eager")

    # Select an audio file and read it:
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    audio_sample = ds[0]["audio"]  # type: ignore

    # Load the Whisper model
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    # Use the model and processor to transcribe the audio:
    input_features = processor(
        audio_sample["array"],  # type: ignore
        sampling_rate=audio_sample["sampling_rate"],  # type: ignore
        return_tensors="pt",
    ).input_features.to(device)

    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )

    # Expected output: "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
    print(transcription[0])


if __name__ == "__main__":
    main()
