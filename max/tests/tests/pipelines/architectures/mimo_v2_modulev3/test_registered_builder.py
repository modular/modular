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
"""Constructs MiMo's registered decoder through the registration machinery.

The role-registration test checks the boundary without building anything;
these checks build the decoder the way a speculative pairing's caller does,
through :class:`RegisteredDecoderBuilder` over the architecture's
registered pipeline model, on the tiny fixture checkpoint and virtual SM100
devices: the plain (non-speculative) build, the selected-layer
(speculative) build, the deployment guards the old pipeline model enforced,
and the verify wiring the DFlash adapter reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.driver import (
    DeviceSpec,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.arch import (
    mimo_v2_modulev3_arch,
)
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import (
    MiMoV2Decoder,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.spec_adapters import (
    MiMoV2BlockTarget,
)
from max.pipelines.architectures.mimo_v2_modulev3.spec_adapters import (
    verify as wire_verify,
)
from max.pipelines.kv_cache import KVCacheConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import RegisteredDecoderBuilder
from max.pipelines.speculative.spec_adapters import VerifyFn
from transformers.configuration_utils import PretrainedConfig

N_DEVICES = 2
"""The tiny fixture's device count: divides both KV groups' heads and the
qkv_proj chunks, so the config's tensor-parallel checks pass."""

BLOCK = 8
"""The DFlash drafter's block width, as the pairing stamps the KV tree."""


@pytest.fixture(scope="module", autouse=True)
def _virtual_sm100() -> None:
    """Virtual SM100 devices, so the config's device validation passes.

    The W4A8 experts' kernels are SM100-only, so the config refuses any
    other device; the virtual devices stand in without a GPU, as the
    integration tests' staging gates do.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture checkpoint written to disk, as the GPU tests do."""
    import tiny_fixture

    path = tmp_path_factory.mktemp("mimo_v3_registered")
    tiny_fixture.write(path, tiny_fixture.tensors())
    return path


@pytest.fixture(scope="module")
def hf(checkpoint: Path) -> PretrainedConfig:
    """The checkpoint's HuggingFace config."""
    return PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )


def _refs(n_devices: int = N_DEVICES) -> list[DeviceRef]:
    return [DeviceRef.GPU(i) for i in range(n_devices)]


def _kv_params(
    hf: PretrainedConfig,
    *,
    speculative: bool = False,
    n_devices: int = N_DEVICES,
) -> MultiKVCacheParams:
    """The ``{sliding, full}`` KV tree, as the config's constructor builds
    it: plain for serving, DFlash-stamped for a speculative pairing."""
    return MiMoV2Config.construct_kv_params(
        hf,
        cast(
            Any,
            SimpleNamespace(
                model=SimpleNamespace(data_parallel_degree=1),
                speculative=(
                    SimpleNamespace(
                        speculative_method="dflash",
                        num_speculative_tokens=BLOCK,
                    )
                    if speculative
                    else None
                ),
            ),
        ),
        _refs(n_devices),
        KVCacheConfig(kv_cache_page_size=128, enable_prefix_caching=False),
        DType.bfloat16,
    )


def _pipeline_config(
    checkpoint: Path,
    *,
    data_parallel_degree: int = 1,
    ep_size: int = 1,
    n_devices: int = N_DEVICES,
) -> Any:
    """A pipeline-config stand-in carrying what the registered build reads:
    the model's device specs, data-parallel degree and HuggingFace config,
    and the runtime's expert-parallel size and batch limits."""
    return SimpleNamespace(
        model=SimpleNamespace(
            data_parallel_degree=data_parallel_degree,
            device_specs=[DeviceSpec.accelerator(i) for i in range(n_devices)],
            huggingface_config=None,
            kv_cache=KVCacheConfig(
                kv_cache_page_size=128, enable_prefix_caching=False
            ),
            # The encoding the config resolves: the checkpoint's own.
            quantization_encoding="float4_e2m1fnx2",
            # The checkpoint's safetensors file, so the given encoding
            # resolves as it does against a real repo: the file name
            # carries no encoding hint, and the file exists, so the
            # requested encoding stands.
            weight_path=[checkpoint / "model.safetensors"],
            huggingface_weight_repo=None,
            default_device_spec=DeviceSpec.accelerator(0),
        ),
        runtime=SimpleNamespace(
            max_batch_input_tokens=8,
            ep_size=ep_size,
            ep_use_allreduce=True,
            enable_chunked_prefill=True,
        ),
        speculative=None,
    )


def _load_context(
    hf: PretrainedConfig,
    checkpoint: Path,
    *,
    data_parallel_degree: int = 1,
    ep_size: int = 1,
    n_devices: int = N_DEVICES,
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    target_layer_ids: tuple[int, ...] = (),
    kv_params: MultiKVCacheParams | None = None,
) -> ModelLoadContext:
    """A load context carrying what the registered builder reads.

    The KV params are the ones a speculative pairing's caller passes: the
    tree the memory planner sized, stamped with the draft's method and
    width. ``None`` derives them from the config, as serving does.
    """
    pipeline_config = _pipeline_config(
        checkpoint,
        data_parallel_degree=data_parallel_degree,
        ep_size=ep_size,
        n_devices=n_devices,
    )
    pipeline_config.model.huggingface_config = hf
    return cast(
        ModelLoadContext,
        SimpleNamespace(
            pipeline_config=pipeline_config,
            weights=object(),
            adapter=None,
            memory_plan=SimpleNamespace(planned_max_batch_size=5),
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=return_hidden_states,
            target_layer_ids=target_layer_ids,
            kv_params=kv_params or _kv_params(hf, n_devices=n_devices),
            max_seq_len=256,
            max_batch_total_tokens=512,
            huggingface_config=hf,
            start_ep=lambda ep: ep,
        ),
    )


@pytest.fixture(autouse=True)
def _no_checkpoint_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The builder reads the checkpoint only to name the decoder's weights,
    which these construction checks don't load; an empty state dict keeps
    the build about the module, not the tensors."""
    monkeypatch.setattr(
        "max.pipelines.lib.module_registration.parse_state_dict_from_weights",
        lambda *args: cast(dict[str, Any], {}),
    )


def _builder() -> RegisteredDecoderBuilder:
    """The builder over the architecture's registered pipeline model."""
    model = mimo_v2_modulev3_arch.pipeline_model
    assert isinstance(model, type)
    return RegisteredDecoderBuilder(cast(Any, model))


def test_the_registered_builder_builds_the_plain_decoder(
    hf: PretrainedConfig,
    checkpoint: Path,
) -> None:
    """The non-speculative build: a ``MiMoV2Decoder`` whose config took.

    The registered path finalizes the config with the caller's output
    policy, so a plain build keeps ``ReturnHiddenStates.NONE`` and the
    last-token logits the sampler reads.
    """
    module, config = _builder().build_module(_load_context(hf, checkpoint))
    assert isinstance(module, MiMoV2Decoder)
    assert isinstance(config, MiMoV2Config)
    assert config.return_hidden_states is ReturnHiddenStates.NONE
    assert config.return_logits is ReturnLogits.LAST_TOKEN
    assert config.data_parallel_degree == 1
    assert config.ep_config is None
    # The decoder the registered role serves: the graph name the
    # precompiled MEFs and Mach's registry bind.
    assert module.graph_name == "MiMoV2"


def test_the_registered_builder_builds_the_speculative_decoder(
    hf: PretrainedConfig,
    checkpoint: Path,
) -> None:
    """The selected-layer build: the decoder initialized the speculative
    state the DFlash adapter's verify pass reads.

    A speculative pairing hands the builder the planner-sized KV tree and
    the drafter's target layers; the built decoder widens its outputs to
    the verify pass's variable logits and captures its taps at those
    layers.
    """
    kv = _kv_params(hf, speculative=True)
    module, config = _builder().build_module(
        _load_context(
            hf,
            checkpoint,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            target_layer_ids=(1, 3),
            kv_params=kv,
        )
    )
    assert isinstance(module, MiMoV2Decoder)
    assert config.target_layer_ids == [1, 3]
    assert module.target_layer_ids == [1, 3]
    # The verify pass's state: variable logits, selected-layer taps.
    assert module.return_logits is ReturnLogits.VARIABLE
    assert module.return_hidden_states is ReturnHiddenStates.SELECTED_LAYERS
    # The KV tree the caller sized is the one the decoder runs on.
    assert config.kv_params is kv


def test_the_registered_builder_refuses_a_dp_tp_mesh(
    hf: PretrainedConfig,
    checkpoint: Path,
) -> None:
    """A dp x tp mesh -- 2 replicas over 4 devices -- is refused by the
    registered config path, as the old pipeline model refused it.

    The refusal fires in the builder's data-parallel check, before the
    runtime is built: a degree that is not the device count leaves no
    one-device replica per device. The KV tree and the virtual device
    count both span the four devices, so the config's device validation
    and the builder's device load see the deployment the refusal is
    about.
    """
    set_virtual_device_count(4)
    try:
        with pytest.raises(ValueError, match="one replica per"):
            _builder().build_module(
                _load_context(
                    hf,
                    checkpoint,
                    data_parallel_degree=2,
                    ep_size=4,
                    n_devices=4,
                )
            )
    finally:
        set_virtual_device_count(N_DEVICES)


def test_the_registered_builder_refuses_dp_without_ep(
    hf: PretrainedConfig,
    checkpoint: Path,
) -> None:
    """DP attention without expert parallelism over every device is
    refused by the registered config path.

    The refusal fires in ``apply_runtime``, which the builder calls
    after the runtime is built: without expert parallelism the EP size
    covers none of the devices the replicas split the batch over.
    """
    with pytest.raises(ValueError, match="one replica per"):
        _builder().build_module(
            _load_context(hf, checkpoint, data_parallel_degree=2, ep_size=1)
        )


def test_the_registered_builder_refuses_tp_spec_ep(
    hf: PretrainedConfig,
    checkpoint: Path,
) -> None:
    """A TP speculative build with expert parallelism is refused: the
    canonical spec-decode signature has no slot for the EP buffers.

    The refusal fires in ``apply_model_load_context``, which the builder
    calls after the config is finalized, before the runtime is built.
    """
    kv = _kv_params(hf, speculative=True)
    with pytest.raises(NotImplementedError, match="requires DP attention"):
        _builder().build_module(
            _load_context(
                hf,
                checkpoint,
                ep_size=N_DEVICES,
                return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
                target_layer_ids=(1, 3),
                kv_params=kv,
            )
        )


def test_the_built_decoder_verifies_through_the_dflash_adapter(
    hf: PretrainedConfig,
    checkpoint: Path,
) -> None:
    """The selected-layer build wires into the DFlash adapter's verify.

    The adapter's ``verify`` takes the built decoder through its target
    type union and returns the block target whose verify pass the pairing
    drives; the drafter's config names the sampleable vocabulary the
    acceptance samples from.
    """
    kv = _kv_params(hf, speculative=True)
    module, config = _builder().build_module(
        _load_context(
            hf,
            checkpoint,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            target_layer_ids=(1, 3),
            kv_params=kv,
        )
    )
    assert isinstance(module, MiMoV2Decoder)
    draft_config = SimpleNamespace(sampleable_vocab_size=448)
    wired: VerifyFn = wire_verify(
        module, config, ReturnHiddenStates.SELECTED_LAYERS, draft_config
    )
    assert isinstance(wired, MiMoV2BlockTarget)
    assert wired.target is module
    assert wired.vocab_size == config.vocab_size
    assert wired.sampleable_vocab_size == 448
