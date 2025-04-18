# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import load_dataset
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops
from max.nn import Conv1D, Embedding, LayerNorm, Linear, Sequential
from max.pipelines.architectures.whisper.encoder import (
    WhisperEncoder,
    WhisperEncoderLayer,
    WhisperSdpaAttention,
)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


@pytest.fixture
def model_id():
    return "openai/whisper-large-v3"


@pytest.fixture
def torch_inputs(model_id):
    """Returns 2 audio files in a tensor of shape = (batch_size=2, n_features=128, seq_length=3000)"""
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    audio_samples = [
        ds[0]["audio"]["array"],  # type: ignore
        ds[1]["audio"]["array"],  # type: ignore
    ]

    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor(
        audio_samples,
        return_attention_mask=True,
        sampling_rate=ds[0]["audio"]["sampling_rate"],  # type: ignore
        return_tensors="pt",
    )

    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    return input_features


@pytest.fixture
def graph_api_inputs(torch_inputs):
    """Returns 2 audio files in a tensor of shape = (batch_size=2, seq_length=3000, n_features=128)"""
    return torch.permute(torch_inputs, (0, 2, 1)).contiguous()


def graph_api_whisper_encoder(weights_registry, model):
    graph_api_conv1 = Conv1D(
        filter=Weight(
            "conv1_weight",
            DType.from_numpy(weights_registry["conv1_weight"].numpy().dtype),
            shape=weights_registry["conv1_weight"].shape,
            device=DeviceRef.CPU(),
        ),
        stride=1,
        padding=1,
        bias=Weight(
            "conv1_bias",
            DType.from_numpy(weights_registry["conv1_bias"].numpy().dtype),
            shape=weights_registry["conv1_bias"].shape,
            device=DeviceRef.CPU(),
        ),
    )
    graph_api_conv2 = Conv1D(
        filter=Weight(
            "conv2_weight",
            DType.from_numpy(weights_registry["conv2_weight"].numpy().dtype),
            shape=weights_registry["conv2_weight"].shape,
            device=DeviceRef.CPU(),
        ),
        stride=2,
        padding=1,
        bias=Weight(
            "conv2_bias",
            DType.from_numpy(weights_registry["conv2_bias"].numpy().dtype),
            shape=weights_registry["conv2_bias"].shape,
            device=DeviceRef.CPU(),
        ),
    )
    graph_api_embed_positions = Embedding(
        Weight(
            "embed_positions",
            DType.from_numpy(weights_registry["embed_positions"].numpy().dtype),
            shape=weights_registry["embed_positions"].shape,
            device=DeviceRef.CPU(),
        ),
    )
    layers = [
        WhisperEncoderLayer(
            attention=WhisperSdpaAttention(
                n_heads=model.model.encoder.layers[i].self_attn.num_heads,
                head_dim=model.model.encoder.layers[i].self_attn.head_dim,
                wq=Linear(
                    Weight(
                        f"layer{i}_self_att_wq",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wq"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[f"layer{i}_self_att_wq"].shape,
                        device=DeviceRef.CPU(),
                    ),
                    bias=Weight(
                        f"layer{i}_self_att_wq_bias",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wq_bias"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[
                            f"layer{i}_self_att_wq_bias"
                        ].shape,
                        device=DeviceRef.CPU(),
                    ),
                ),
                wk=Linear(
                    Weight(
                        f"layer{i}_self_att_wk",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wk"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[f"layer{i}_self_att_wk"].shape,
                        device=DeviceRef.CPU(),
                    )
                ),
                wv=Linear(
                    Weight(
                        f"layer{i}_self_att_wv",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wv"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[f"layer{i}_self_att_wv"].shape,
                        device=DeviceRef.CPU(),
                    ),
                    bias=Weight(
                        f"layer{i}_self_att_wv_bias",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wv_bias"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[
                            f"layer{i}_self_att_wv_bias"
                        ].shape,
                        device=DeviceRef.CPU(),
                    ),
                ),
                wo=Linear(
                    Weight(
                        f"layer{i}_self_att_wo",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wo"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[f"layer{i}_self_att_wo"].shape,
                        device=DeviceRef.CPU(),
                    ),
                    bias=Weight(
                        f"layer{i}_self_att_wo_bias",
                        DType.from_numpy(
                            weights_registry[f"layer{i}_self_att_wo_bias"]
                            .numpy()
                            .dtype
                        ),
                        shape=weights_registry[
                            f"layer{i}_self_att_wo_bias"
                        ].shape,
                        device=DeviceRef.CPU(),
                    ),
                ),
            ),
            mlp=Sequential(
                layers=[
                    Linear(
                        Weight(
                            f"layer{i}_fc1",
                            DType.from_numpy(
                                weights_registry[f"layer{i}_fc1"].numpy().dtype
                            ),
                            shape=weights_registry[f"layer{i}_fc1"].shape,
                            device=DeviceRef.CPU(),
                        ),
                        bias=Weight(
                            f"layer{i}_fc1_bias",
                            DType.from_numpy(
                                weights_registry[f"layer{i}_fc1_bias"]
                                .numpy()
                                .dtype
                            ),
                            shape=weights_registry[f"layer{i}_fc1_bias"].shape,
                            device=DeviceRef.CPU(),
                        ),
                    ),
                    ops.gelu,  # type: ignore
                    Linear(
                        Weight(
                            f"layer{i}_fc2",
                            DType.from_numpy(
                                weights_registry[f"layer{i}_fc2"].numpy().dtype
                            ),
                            shape=weights_registry[f"layer{i}_fc2"].shape,
                            device=DeviceRef.CPU(),
                        ),
                        bias=Weight(
                            f"layer{i}_fc2_bias",
                            DType.from_numpy(
                                weights_registry[f"layer{i}_fc2_bias"]
                                .numpy()
                                .dtype
                            ),
                            shape=weights_registry[f"layer{i}_fc2_bias"].shape,
                            device=DeviceRef.CPU(),
                        ),
                    ),
                ]
            ),
            attention_norm=LayerNorm(
                weight=Weight(
                    f"layer{i}_attention_norm",
                    DType.from_numpy(
                        weights_registry[f"layer{i}_attention_norm"]
                        .numpy()
                        .dtype
                    ),
                    shape=weights_registry[f"layer{i}_attention_norm"].shape,
                    device=DeviceRef.CPU(),
                ),
                eps=1e-5,
                bias=Weight(
                    f"layer{i}_attention_norm_bias",
                    DType.from_numpy(
                        weights_registry[f"layer{i}_attention_norm_bias"]
                        .numpy()
                        .dtype
                    ),
                    shape=weights_registry[
                        f"layer{i}_attention_norm_bias"
                    ].shape,
                    device=DeviceRef.CPU(),
                ),
            ),
            mlp_norm=LayerNorm(
                weight=Weight(
                    f"layer{i}_mlp_norm",
                    DType.from_numpy(
                        weights_registry[f"layer{i}_mlp_norm"].numpy().dtype
                    ),
                    shape=weights_registry[f"layer{i}_mlp_norm"].shape,
                    device=DeviceRef.CPU(),
                ),
                eps=1e-5,
                bias=Weight(
                    f"layer{i}_mlp_norm_bias",
                    DType.from_numpy(
                        weights_registry[f"layer{i}_mlp_norm_bias"]
                        .numpy()
                        .dtype
                    ),
                    shape=weights_registry[f"layer{i}_mlp_norm_bias"].shape,
                    device=DeviceRef.CPU(),
                ),
            ),
        )
        for i in range(len(model.model.encoder.layers))
    ]
    norm = LayerNorm(
        weight=Weight(
            "final_norm",
            DType.from_numpy(weights_registry["final_norm"].numpy().dtype),
            shape=weights_registry["final_norm"].shape,
            device=DeviceRef.CPU(),
        ),
        eps=1e-5,
        bias=Weight(
            "final_norm_bias",
            DType.from_numpy(weights_registry["final_norm_bias"].numpy().dtype),
            shape=weights_registry["final_norm_bias"].shape,
            device=DeviceRef.CPU(),
        ),
    )
    return WhisperEncoder(
        graph_api_conv1,
        graph_api_conv2,
        graph_api_embed_positions,
        layers,
        norm,
    )


@pytest.mark.skip(
    reason="We decided to postpone finishing Whisper bring up. Should debug if we come back to it."
)
def test_encoder_stem(torch_inputs, graph_api_inputs, model_id):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

    # inputs = np.load("/home/ubuntu/modular/whisper_sample2.npy")

    inputs_embeds = torch.nn.functional.gelu(
        model.model.encoder.conv1(torch_inputs)
    )
    inputs_embeds = torch.nn.functional.gelu(
        model.model.encoder.conv2(inputs_embeds)
    )

    torch_inputs_embeds = torch.permute(inputs_embeds, (0, 2, 1))
    embed_pos = (
        model.model.encoder.embed_positions.weight
    )  # shape = [1500, 1280]
    torch_hidden_states = (
        torch_inputs_embeds + embed_pos
    )  # shape = [2, 1500, 1280]

    # graph_api weights
    weights_registry = {}
    weights_registry["conv1_weight"] = torch.permute(
        model.model.encoder.conv1.weight.data, (2, 1, 0)
    ).contiguous()  # [kernel_size, in_channels, out_channels]
    weights_registry["conv2_weight"] = torch.permute(
        model.model.encoder.conv2.weight.data, (2, 1, 0)
    ).contiguous()  # [kernel_size, in_channels, out_channels]
    weights_registry["conv1_bias"] = (
        model.model.encoder.conv1.bias.data.contiguous()
    )  # [out_channels]
    weights_registry["conv2_bias"] = (
        model.model.encoder.conv2.bias.data.contiguous()
    )  # [out_channels]
    weights_registry["embed_positions"] = (
        model.model.encoder.embed_positions.weight.data
    )  # [out_channels]

    graph_api_conv1 = Conv1D(
        filter=Weight(
            "conv1_weight",
            DType.from_numpy(weights_registry["conv1_weight"].numpy().dtype),
            shape=weights_registry["conv1_weight"].shape,
            device=DeviceRef.CPU(),
        ),
        stride=1,
        padding=1,
        bias=Weight(
            "conv1_bias",
            DType.from_numpy(weights_registry["conv1_bias"].numpy().dtype),
            shape=weights_registry["conv1_bias"].shape,
            device=DeviceRef.CPU(),
        ),
    )
    graph_api_conv2 = Conv1D(
        filter=Weight(
            "conv2_weight",
            DType.from_numpy(weights_registry["conv2_weight"].numpy().dtype),
            shape=weights_registry["conv2_weight"].shape,
            device=DeviceRef.CPU(),
        ),
        stride=2,
        padding=1,
        bias=Weight(
            "conv2_bias",
            DType.from_numpy(weights_registry["conv2_bias"].numpy().dtype),
            shape=weights_registry["conv2_bias"].shape,
            device=DeviceRef.CPU(),
        ),
    )

    graph_api_embed_positions = Embedding(
        Weight(
            "embed_positions",
            DType.from_numpy(weights_registry["embed_positions"].numpy().dtype),
            shape=weights_registry["embed_positions"].shape,
            device=DeviceRef.CPU(),
        ),
    )

    session = InferenceSession()
    with Graph(
        name="stem",
        input_types=(
            TensorType(
                DType.from_numpy(graph_api_inputs.numpy().dtype),
                graph_api_inputs.shape,
            ),
        ),
    ) as graph:
        graph_api_inputs_embeds = ops.gelu(
            graph_api_conv1(graph.inputs[0].tensor)
        )
        graph_api_inputs_embeds = ops.gelu(
            graph_api_conv2(graph_api_inputs_embeds)
        )
        graph_api_hidden_states = (
            graph_api_inputs_embeds + graph_api_embed_positions.weights
        )
        graph.output(graph_api_hidden_states)

    compiled = session.load(graph, weights_registry=weights_registry)

    graph_api_hidden_states = compiled.execute(graph_api_inputs)[0]
    assert isinstance(graph_api_hidden_states, Tensor)

    np.testing.assert_allclose(
        graph_api_hidden_states.to_numpy(),
        torch_hidden_states.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


@pytest.mark.skip(
    reason="We decided to postpone finishing Whisper bring up. Should debug if we come back to it."
)
def test_whisper_encoder(torch_inputs, graph_api_inputs, model_id):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

    torch_outputs = model.model.encoder(torch_inputs).last_hidden_state

    # graph_api weights
    weights_registry = {}
    weights_registry["conv1_weight"] = torch.permute(
        model.model.encoder.conv1.weight.data, (2, 1, 0)
    ).contiguous()  # [kernel_size, in_channels, out_channels]
    weights_registry["conv2_weight"] = torch.permute(
        model.model.encoder.conv2.weight.data, (2, 1, 0)
    ).contiguous()  # [kernel_size, in_channels, out_channels]
    weights_registry["conv1_bias"] = (
        model.model.encoder.conv1.bias.data.contiguous()
    )  # [out_channels]
    weights_registry["conv2_bias"] = (
        model.model.encoder.conv2.bias.data.contiguous()
    )  # [out_channels]
    weights_registry["embed_positions"] = (
        model.model.encoder.embed_positions.weight.data
    )  # [out_channels]
    weights_registry["embed_positions"] = (
        model.model.encoder.embed_positions.weight.data
    )
    weights_registry["final_norm"] = model.model.encoder.layer_norm.weight.data
    weights_registry["final_norm_bias"] = (
        model.model.encoder.layer_norm.weight.data
    )
    for i in range(len(model.model.encoder.layers)):
        weights_registry[f"layer{i}_mlp_norm"] = model.model.encoder.layers[
            i
        ].final_layer_norm.weight.data
        weights_registry[f"layer{i}_mlp_norm_bias"] = (
            model.model.encoder.layers[i].final_layer_norm.bias.data
        )
        weights_registry[f"layer{i}_attention_norm"] = (
            model.model.encoder.layers[i].self_attn_layer_norm.weight.data
        )
        weights_registry[f"layer{i}_attention_norm_bias"] = (
            model.model.encoder.layers[i].self_attn_layer_norm.bias.data
        )
        weights_registry[f"layer{i}_fc1"] = model.model.encoder.layers[
            i
        ].fc1.weight.data
        weights_registry[f"layer{i}_fc1_bias"] = model.model.encoder.layers[
            i
        ].fc1.bias.data
        weights_registry[f"layer{i}_fc2"] = model.model.encoder.layers[
            i
        ].fc2.weight.data
        weights_registry[f"layer{i}_fc2_bias"] = model.model.encoder.layers[
            i
        ].fc2.bias.data
        weights_registry[f"layer{i}_self_att_wo"] = model.model.encoder.layers[
            i
        ].self_attn.out_proj.weight.data
        weights_registry[f"layer{i}_self_att_wo_bias"] = (
            model.model.encoder.layers[i].self_attn.out_proj.bias.data
        )
        weights_registry[f"layer{i}_self_att_wq"] = model.model.encoder.layers[
            i
        ].self_attn.q_proj.weight.data
        weights_registry[f"layer{i}_self_att_wq_bias"] = (
            model.model.encoder.layers[i].self_attn.q_proj.bias.data
        )
        weights_registry[f"layer{i}_self_att_wk"] = model.model.encoder.layers[
            i
        ].self_attn.k_proj.weight.data
        weights_registry[f"layer{i}_self_att_wv"] = model.model.encoder.layers[
            i
        ].self_attn.v_proj.weight.data
        weights_registry[f"layer{i}_self_att_wv_bias"] = (
            model.model.encoder.layers[i].self_attn.v_proj.bias.data
        )

    graph_api_model = graph_api_whisper_encoder(weights_registry, model)

    session = InferenceSession()
    with Graph(
        name="encoder",
        input_types=(
            TensorType(
                DType.from_numpy(graph_api_inputs.numpy().dtype),
                graph_api_inputs.shape,
            ),
        ),
    ) as graph:
        graph_api_output = graph_api_model(graph.inputs[0].tensor)[0]
        graph.output(graph_api_output)

    compiled = session.load(graph, weights_registry=weights_registry)

    graph_api_output = compiled.execute(graph_api_inputs)[0]
    assert isinstance(graph_api_output, Tensor)

    np.testing.assert_allclose(
        graph_api_output.to_numpy(),
        torch_outputs.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
