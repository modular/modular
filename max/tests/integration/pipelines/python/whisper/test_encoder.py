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
from max.graph import Graph, TensorType, Weight, ops
from max.pipelines.nn import Conv1D
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


@pytest.fixture
def audio() -> tuple[list[np.ndarray], int]:
    """Returns 2 audio files in a tensor of shape = (batch_size=2, )"""
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    audio_samples = [
        ds[0]["audio"]["array"],  # type: ignore
        ds[1]["audio"]["array"],  # type: ignore
    ]
    print(ds[0]["audio"]["sampling_rate"])  # type: ignore
    print(ds[1]["audio"]["sampling_rate"])  # type: ignore
    # assert ds[0]["audio"]["sampling_rate"] == ds[1]["audio"]["sampling_rate"] # type: ignore
    return audio_samples, ds[0]["audio"]["sampling_rate"]  # type: ignore


@pytest.fixture
def model_and_processor() -> tuple[
    WhisperForConditionalGeneration, WhisperProcessor
]:
    model_id = "openai/whisper-large-v3"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    return model, processor


@pytest.fixture
def graph_api_whisper_encoder(weights_registry):
    conv1 = Conv1D(
        filter=Weight(
            "conv1",
            DType.from_numpy(weights_registry["conv1"].numpy().dtype),
            shape=weights_registry["conv1"].shape,
        ),
        stride=1,
        padding=1,
    )
    conv2 = Conv1D(
        filter=Weight(
            "conv2",
            DType.from_numpy(weights_registry["conv2"].numpy().dtype),
            shape=weights_registry["conv2"].shape,
        ),
        stride=2,
        padding=1,
    )


@pytest.mark.skip(
    reason="Initial test components are added but the layers are not implemented yet."
)
def test_whisper_encoder(audio, model_and_processor):
    audio_samples, sampling_rate = audio

    torch_model, processor = model_and_processor

    # (batch_size, feature_size, sequence_length) [2, 128, 3000]
    torch_input_features = processor(
        audio_samples, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    # [2, 128, 3000]
    torch_input_features = torch_model.model._mask_input_features(
        torch_input_features, attention_mask=None
    )

    # encoder output shape = torch.Size([1, 1500, 1280])


@pytest.mark.skip(
    reason="Initial test components are added but the layers are not implemented yet."
)
def test_encoder_stem():
    model_id = "openai/whisper-large-v3"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

    sampling_rate = processor.feature_extractor.sampling_rate
    inputs = np.load("/home/ubuntu/modular/whisper_sample2.npy")

    inputs = processor(
        inputs,
        return_attention_mask=True,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs["input_features"]

    inputs_embeds = torch.nn.functional.gelu(
        model.model.encoder.conv1(input_features)
    )
    inputs_embeds = torch.nn.functional.gelu(
        model.model.encoder.conv2(inputs_embeds)
    )

    inputs_embeds = torch.permute(inputs_embeds, (0, 2, 1))

    # graph_api
    graph_api_input_features = torch.permute(
        input_features, (0, 2, 1)
    ).contiguous()

    weights_registry = {}
    weights_registry["conv1"] = torch.permute(
        model.model.encoder.conv1.weight.data, (2, 1, 0)
    ).contiguous()  # [kernel_size, in_channels, out_channels]
    weights_registry["conv2"] = torch.permute(
        model.model.encoder.conv2.weight.data, (2, 1, 0)
    ).contiguous()  # [kernel_size, in_channels, out_channels]

    graph_api_conv1 = Conv1D(
        filter=Weight(
            "conv1",
            DType.from_numpy(weights_registry["conv1"].numpy().dtype),
            shape=weights_registry["conv1"].shape,
        ),
        stride=1,
        padding=1,
    )
    graph_api_conv2 = Conv1D(
        filter=Weight(
            "conv2",
            DType.from_numpy(weights_registry["conv2"].numpy().dtype),
            shape=weights_registry["conv2"].shape,
        ),
        stride=2,
        padding=1,
    )

    session = InferenceSession()
    with Graph(
        name="stem",
        input_types=(
            TensorType(
                DType.from_numpy(graph_api_input_features.numpy().dtype),
                graph_api_input_features.shape,
            ),
        ),
    ) as graph:
        graph_api_inputs_embeds = ops.gelu(
            graph_api_conv1(graph.inputs[0].tensor)
        )
        graph_api_inputs_embeds = ops.gelu(
            graph_api_conv2(graph_api_inputs_embeds)
        )
        graph.output(graph_api_inputs_embeds)

    compiled = session.load(graph, weights_registry=weights_registry)

    graph_api_outputs = compiled.execute(graph_api_input_features)[0]
    assert isinstance(graph_api_outputs, Tensor)

    np.testing.assert_allclose(
        graph_api_outputs.to_numpy(),
        inputs_embeds.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
