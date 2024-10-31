# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
from nn.conv import Conv2D
from nn.linear import Linear
from nn.norm import RMSNorm
from pixtral.model.attention import Attention
from pixtral.model.attention_utils import causal_attention_mask_2d
from pixtral.model.rotary_embedding_2d import (
    RotaryEmbedding2D,
    patch_position_ids,
)
from pixtral.model.transformer import MLP, Transformer, TransformerBlock
from torch import Tensor, nn

num_channels = 3  # Number of input channels in the input images.
hidden_size = 1024  # Dimension of the hidden representations.
intermediate_size = 4096  # Dimension of the MLP representations.
patch_size = 16  # Size of the image patches.
rope_theta = 10000.0  # The base period of the RoPE embeddings.
num_attention_heads = (
    16  #  Number of attention heads in the Transformer encoder.
)
head_dim = (
    hidden_size // num_attention_heads
)  # dim of positional embeddings and attention heads
scale = head_dim**-0.5
num_hidden_layers = 24  # Number of hidden layers in the Transformer encoder.
hidden_act = "gelu"  # Activation function used in the hidden layers.
image_size = 1024  # Max dimension of the input images. Should be 1024
attention_dropout = 0.0  # Dropout probability for the attention layers.
variance_epsilon = 1e-5


# Arrange
@pytest.fixture
def img_sizes():
    return [(128, 64), (128, 256)]


@pytest.fixture
def img_dtype():
    return torch.float32


@pytest.fixture
def imgs(img_sizes, img_dtype):
    # generate imgs of shape (batch_size=1, num_channels, height, width)
    return [
        torch.randint(low=0, high=255, size=(num_channels, height, width)).to(
            img_dtype
        )
        for height, width in img_sizes
    ]


# TODO(KERN-1066): Fix and enable test
@pytest.mark.skip(reason="Errors are larger than usual (10^-2)")
def test_patch_conv(imgs, img_sizes):
    # TODO: Check the values of pixels are expected to be in [0, 255]
    # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/pixtral/modeling_pixtral.py#L465
    patch_conv = nn.Conv2d(
        in_channels=num_channels,
        out_channels=hidden_size,
        kernel_size=patch_size,
        stride=patch_size,
        bias=False,
    )

    with torch.no_grad():
        filters = patch_conv.weight.data
        imgs = [img.unsqueeze(0) for img in imgs]
        patch_embeds_list = [patch_conv(img) for img in imgs]

        patch_embeds_list = [
            torch.permute(img, (0, 2, 3, 1)) for img in patch_embeds_list
        ]
        filters = torch.permute(filters, (2, 3, 1, 0))

        graph_api_imgs = [torch.permute(img, (0, 2, 3, 1)) for img in imgs]

        session = InferenceSession()
        graph = Graph(
            "conv",
            Conv2D(filters.numpy(), stride=(patch_size, patch_size)),
            input_types=(
                TensorType(
                    DType.float32, (1, "img_height", "img_width", num_channels)
                ),
            ),
        )

        compiled = session.load(graph)

        output = [
            compiled.execute(np.ascontiguousarray(img))[0].to_numpy()
            for img in graph_api_imgs
        ]

        ACCURACY_RTOL = 1e-4
        ACCURACY_ATOL = 1e-6
        np.testing.assert_allclose(
            output[1],
            patch_embeds_list[1].detach().numpy(),
            equal_nan=True,
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )


class PixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


class PixtralRotaryEmbedding(nn.Module):
    """
    The key with pixtral embedding is just that you have a frequency for each pixel positions.
    If you have height x width pixels (or embedding pixels), then the frequency used for ROPE
    is given by indexing the pre_computed frequency on the width and height.

    What you output is of dimension (batch, height * width, dim) with dim the embed dim.

    This simply means that for each image hidden state, you are going to add
    a corresponding positional embedding, based on its index in the grid.
    """

    def __init__(self, device):
        super().__init__()
        self.rope_type = "default"
        self.dim = head_dim  # 64
        self.base = rope_theta
        max_patches_per_side = image_size // patch_size  # 64

        # 1D tensor of length head_dim // 2 = 32
        freqs = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        # 1D tensor of length max_patches_per_side = 64
        h = torch.arange(max_patches_per_side, device=freqs.device)
        # 1D tensor of length max_patches_per_side = 64
        w = torch.arange(max_patches_per_side, device=freqs.device)

        # 2D tensors of shape (max_patches_per_side = 64, len(freqs)//2 =16)
        freqs_h = torch.outer(h, freqs[::2]).float()
        freqs_w = torch.outer(w, freqs[1::2]).float()

        # 2D tensor of shape (max_patches_per_side*max_patches_per_side = 4096,  head_dim // 2 = 32)
        inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
            ],
            dim=-1,
        ).reshape(
            -1, self.dim // 2
        )  # we reshape to only index on the position indexes, not tuple of indexes
        # Different from paper, but it uses a different permutation in order to obtain the same calculation

        # TODO maybe make it torch compatible later on. We can also just slice
        # 2D tensor of shape (max_patches_per_side*max_patches_per_side =4096, head_dim=64)
        self.register_buffer(
            "inv_freq",
            torch.cat((inv_freq, inv_freq), dim=-1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        # 2D tensor of shape (actual num_patches, head_dim=64)
        freqs = self.inv_freq[position_ids]
        # position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str)
            and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            emb = freqs
            # 2D tensors of shape (actual num_patches, head_dim=64)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer(
                "inv_freq", self.original_inv_freq, persistent=False
            )
            self.max_seq_len_cached = self.original_max_seq_len


def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    d_min = -10000.0
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device
    )

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(
        tensor.shape[0], 1, -1, -1
    )
    return causal_mask


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
}

ACT2FN = ClassInstantier(ACT2CLS)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Pixtral
class PixtralMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state))
            * self.up_proj(hidden_state)
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PixtralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, patches, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings  # type:ignore
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=0
        )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, patches, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class PixtralAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP()
        self.attention = PixtralAttention()
        self.ffn_norm = PixtralRMSNorm(hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # if output_attentions:
        #    outputs += (attn_weights,)
        return outputs


class PixtralTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(PixtralAttentionLayer())
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embeddings which serve as input to the Transformer.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # encoder_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None
        encoder_states = ()
        all_attentions = ()

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            # if output_hidden_states:
            #    encoder_states = encoder_states + (hidden_states,) # type:ignore
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # if output_attentions:
            #    all_attentions = all_attentions + (layer_outputs[1],)

        # if output_hidden_states:
        #    encoder_states = encoder_states + (hidden_states,)

        # if not return_dict:
        #    return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # return BaseModelOutput(
        #    last_hidden_state=hidden_states, hidden_states=[hidden_states], attentions=all_attentions
        # )
        return tuple(
            v
            for v in [hidden_states, encoder_states, all_attentions]
            if v is not None
        )


PIXTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PixtralVisionConfig`]):
            Model configuration class with all the parameters of the vision encoder. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


def test_pixtral_attention(imgs, img_sizes):
    # TODO: Check the values of pixels are expected to be in [0, 255]
    # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/pixtral/modeling_pixtral.py#L465
    device = "cuda" if torch.cuda.is_available() else "cpu"

    patch_conv = nn.Conv2d(
        in_channels=num_channels,
        out_channels=hidden_size,
        kernel_size=patch_size,
        stride=patch_size,
        bias=False,
    )
    ln_pre = PixtralRMSNorm(hidden_size, eps=1e-5)
    patch_positional_embedding = PixtralRotaryEmbedding(device=device)
    transformer = PixtralTransformer()

    with torch.no_grad():
        filters = patch_conv.weight.data
        # list of size (batch_size, hidden_size, num_height_patches, num_width_patches)
        patch_embeds_list = [patch_conv(img.unsqueeze(0)) for img in imgs]

        # seq_len = sum(num_height_patches*num_width_patches) for all images in imgs
        # flatten to a single sequence of shape (batch_size, seq_len=num_patches, hidden_size)
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1
        )
        norm_patch_embeds = ln_pre(patch_embeds)
        # positional embeddings (seq_len)
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=image_size // patch_size
        ).to(device)
        # a tuple of 2 tensors of shape (seq_len, hidden_size/num_attention_heads=64)
        position_embedding = patch_positional_embedding(
            norm_patch_embeds, position_ids
        )
        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
            norm_patch_embeds,
        )
        # (, hidden_size)
        encoder_output = transformer(
            norm_patch_embeds, attention_mask, position_embedding
        )

        ####### Permute torch inputs for the graph API and init weights ########
        patch_embeds_list = [
            torch.permute(img, (0, 2, 3, 1)) for img in patch_embeds_list
        ]
        filters = torch.permute(filters, (2, 3, 1, 0))
        imgs = [
            np.ascontiguousarray(torch.permute(img, (1, 2, 0))) for img in imgs
        ]
        rms_norm_weight = np.ones(hidden_size)
        mlp_gate_weights = []
        for i in range(num_hidden_layers):
            mlp_gate_weights.append(
                transformer.layers[i].feed_forward.gate_proj.weight.data
            )
        mlp_up_weights = []
        for i in range(num_hidden_layers):
            mlp_up_weights.append(
                transformer.layers[i].feed_forward.up_proj.weight.data
            )
        mlp_down_weights = []
        for i in range(num_hidden_layers):
            mlp_down_weights.append(
                transformer.layers[i].feed_forward.down_proj.weight.data
            )
        attention_k_proj_weights = []
        for i in range(num_hidden_layers):
            attention_k_proj_weights.append(
                transformer.layers[i].attention.k_proj.weight.data
            )
        attention_v_proj_weights = []
        for i in range(num_hidden_layers):
            attention_v_proj_weights.append(
                transformer.layers[i].attention.v_proj.weight.data
            )
        attention_q_proj_weights = []
        for i in range(num_hidden_layers):
            attention_q_proj_weights.append(
                transformer.layers[i].attention.q_proj.weight.data
            )
        attention_o_proj_weights = []
        for i in range(num_hidden_layers):
            attention_o_proj_weights.append(
                transformer.layers[i].attention.o_proj.weight.data
            )

        ########### Graph API Pixtral Layers #########
        session = InferenceSession()
        with Graph(
            "conv",
            input_types=[
                TensorType(DType.float32, (h, w, num_channels))
                for h, w in img_sizes
            ],
        ) as graph:
            graph_inputs = graph.inputs
            graph_patch_conv = Conv2D(
                filters.numpy(), stride=(patch_size, patch_size)
            )
            graph_ln_pre = RMSNorm(weight=rms_norm_weight, eps=1e-5)
            # TODO: max_seq_len should be the max number of patches.
            graph_rope = RotaryEmbedding2D(
                dim=hidden_size,
                n_heads=num_attention_heads,
                theta=rope_theta,
                max_patches_per_side=image_size // patch_size,
            )
            attention_layers = []
            for i in range(num_hidden_layers):
                # TODO: init weights for Linear? should be similar o nn.Linear
                gate_proj = Linear(mlp_gate_weights[i])
                down_proj = Linear(mlp_down_weights[i])
                up_proj = Linear(mlp_up_weights[i])
                mlp = MLP(gate_proj, down_proj, up_proj)
                # TODO: init weights
                wq = Linear(attention_q_proj_weights[i])
                wk = Linear(attention_k_proj_weights[i])
                wv = Linear(attention_v_proj_weights[i])
                wo = Linear(attention_o_proj_weights[i])
                attention = Attention(
                    n_heads=num_attention_heads,
                    dim=hidden_size,
                    head_dim=hidden_size // num_attention_heads,
                    dropout=attention_dropout,
                    wq=wq,
                    wk=wk,
                    wv=wv,
                    wo=wo,
                )
                attention_norm = RMSNorm(weight=np.ones(hidden_size), eps=1e-5)
                mlp_norm = RMSNorm(weight=np.ones(hidden_size), eps=1e-5)
                attention_layers.append(
                    TransformerBlock(attention, mlp, attention_norm, mlp_norm)
                )

            graph_transformer = Transformer(
                num_attention_heads, attention_layers
            )

            # Vision Encoder Code.
            # list of [batch_size, new_height, new_width, hidden_size]
            graph_patch_embeds_list = [
                graph_patch_conv(ops.unsqueeze(img, 0)) for img in graph_inputs
            ]
            # tensor of shape [batch_size, seq_len=num_patches, hidden_size]
            graph_patch_embeds = ops.concat(
                [  # p.shape = batch_size, patches_per_height, patches_per_width, hidden_size
                    p.reshape((p.shape[0], -1, p.shape[3]))
                    for p in graph_patch_embeds_list
                ],
                axis=1,
            )
            norm_graph_patch_embeds = graph_ln_pre(graph_patch_embeds)
            # tensor of shape [seq_len = n_patches]
            graph_position_ids = patch_position_ids(
                patch_embeds_list, max_width=image_size // patch_size
            )
            graph_position_embedding = graph_rope(
                norm_graph_patch_embeds, graph_position_ids
            )

            graph_attention_mask = causal_attention_mask_2d(
                [p.shape[1] * p.shape[2] for p in graph_patch_embeds_list],
                norm_patch_embeds,
            )

            # graph_encoder_output = graph_transformer(norm_graph_patch_embeds, graph_attention_mask, graph_position_embedding)
            graph.output(graph_position_embedding[0])
            compiled = session.load(graph)

            output = compiled.execute(*imgs)[0].to_numpy()

            ACCURACY_RTOL = 1e-4
            ACCURACY_ATOL = 1e-6

            np.testing.assert_allclose(
                output,
                position_embedding[0].detach().numpy(),
                equal_nan=True,
                rtol=ACCURACY_RTOL,
                atol=ACCURACY_ATOL,
            )

            np.testing.assert_allclose(
                graph_attention_mask,
                attention_mask.detach().numpy(),
                equal_nan=True,
                rtol=ACCURACY_RTOL,
                atol=ACCURACY_ATOL,
            )
