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
"""Mistral Large 4's vision pipeline: registration, preprocessing, wiring.

S3 makes the architecture text+vision, so the retired S2 gate
(``test_vision_gating.py``, which pinned the model OUT of the vision drive
until the tokenizer and context landed) is replaced here by the pins that
hold the wiring together:

* the architecture registers the vision tokenizer, the vision context and
  image input, and the pipeline model satisfies the structural
  ``SupportsVisionEncoding`` protocol the pipeline drives -- spec, methods
  and vision context land together, never a spec that nothing can execute;
* the processor grows each image placeholder into rows of image tokens
  closed by break tokens and one end token, sizes like the HF reference
  (shrink past 1540 with floor, round every side up to a token multiple,
  never upscale), and its token layout is exact against the S1 golden
  prompts, through the real tokenizer, wherever the fixture is present;
* ``graph_specs`` returns both graph roles -- the decoder with the
  vision-merge inputs at its ABI tail and the vision tower on device 0 --
  and both trace; a module built without ``encodes_images`` keeps the
  text-only ABI, which is what the fused DSpark graph traces against;
* a batch with no image never enters vision encoding, and the vision
  encoder cache is enabled: the arch config sizes its worst-case entry,
  so the pipeline reserves a budget instead of silently re-encoding;
* the exporter's language-only build compiles the decoder alone with the
  text-only ABI, and the shared input packing keeps the text-only
  buffer layout whenever the vision fields are unset.

The graph_specs trace runs on a virtual B200 (no GPU, no compile); the
seeded model instance traces the production method, so the roles, the ABI
order and the two weight slices are what serving builds.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Device,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV3_modulev3.batch_processor import (
    DeepseekV3ModuleV3BatchProcessor,
)
from max.pipelines.architectures.deepseekV3_modulev3.inputs import (
    DeepseekV3Inputs,
)
from max.pipelines.architectures.mistral_large4_modulev3 import (
    model as ml4_model,
)
from max.pipelines.architectures.mistral_large4_modulev3.arch import (
    mistral_large4_modulev3_arch,
)
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    MistralLarge4Config,
    build_kv_params,
)
from max.pipelines.architectures.mistral_large4_modulev3.context import (
    MistralLarge4VisionContext,
)
from max.pipelines.architectures.mistral_large4_modulev3.mistral_large4 import (
    MistralLarge4,
)
from max.pipelines.architectures.mistral_large4_modulev3.model import (
    MistralLarge4Model,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    MistralLarge4NormalizedConfig,
)
from max.pipelines.architectures.mistral_large4_modulev3.processor import (
    MistralLarge4Processor,
)
from max.pipelines.architectures.mistral_large4_modulev3.tokenizer import (
    MistralLarge4VisionTokenizer,
)
from max.pipelines.architectures.mistral_large4_modulev3.vision import (
    MistralLarge4Vision,
    MistralLarge4VisionConfig,
    vision_input_types,
)
from max.pipelines.architectures.mistral_large4_modulev3.weight_contract import (
    VISION_PREFIXES,
)
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces import arch_has_vision_tower
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.memory_plan import _vision_row, vision_cache_plan
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.lib.vision_encoder_cache import (
    SupportsVisionEncoding,
    VisionEncoderCache,
)
from max.pipelines.modeling.types import InputModality
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from safetensors import safe_open
from transformers import PretrainedConfig

_TESTDATA = Path("max/tests/integration/architectures/mistral_large4/testdata")
FIXTURE_ROOT = Path(
    os.environ.get("M4_FIXTURE_ROOT", "/home/ubuntu/m4-fixture")
)
MODELS_ROOT = Path(os.environ.get("M4_MODELS_ROOT", "/home/ubuntu/models"))
GOLDEN = FIXTURE_ROOT / "golden" / "vision" / "bf16"
CKPT = MODELS_ROOT / "Mistral-Large-4-mini-NVFP4"

# The processor constants, as the checkpoint's processor_config.json spells
# them: one image token covers a merged 2x2 square of 14-pixel patches.
PROCESSOR_CONFIG = {
    "image_token": "[IMG]",
    "image_break_token": "[IMG_BREAK]",
    "image_end_token": "[IMG_END]",
    "patch_size": 14,
    "spatial_merge_size": 2,
    "image_processor": {
        "size": {"longest_edge": 1540},
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
}

# The checkpoint's vision geometry, as test_vision_graph pins it against the
# committed headers.
VISION_CONFIG = MistralLarge4VisionConfig(
    dtype=DType.bfloat16,
    num_channels=3,
    patch_size=14,
    image_size=1540,
    vision_hidden_size=2048,
    vision_num_attention_heads=16,
    vision_head_dim=128,
    vision_rope_theta=10000.0,
    vision_num_hidden_layers=30,
    vision_intermediate_size=6144,
    spatial_merge_size=2,
    hidden_size=6144,
    rms_norm_eps=1e-5,
)

_TOKEN_IDS = {"[IMG]": 10, "[IMG_BREAK]": 12, "[IMG_END]": 13}


class _StubDelegate:
    """The token ids a processor needs, without a tokenizer behind them."""

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [_TOKEN_IDS[token] for token in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        # One token per character, with the placeholder tokens recognized
        # whole, as a real tokenizer maps them.
        ids: list[int] = []
        i = 0
        while i < len(text):
            for token, token_id in _TOKEN_IDS.items():
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]) % 256)
                i += 1
        return ids


def _processor() -> MistralLarge4Processor:
    return MistralLarge4Processor(_StubDelegate(), PROCESSOR_CONFIG)


def test_the_architecture_registers_the_vision_pipeline() -> None:
    """The arch serves text and image through the vision tokenizer."""
    assert mistral_large4_modulev3_arch.input_modalities == {
        InputModality.TEXT,
        InputModality.IMAGE,
    }
    assert (
        mistral_large4_modulev3_arch.tokenizer is MistralLarge4VisionTokenizer
    )
    assert (
        mistral_large4_modulev3_arch.context_type is MistralLarge4VisionContext
    )
    assert issubclass(MistralLarge4VisionContext, TextAndVisionContext)
    # The structural protocol both text-generation variants drive: with the
    # vision graph spec, the entry points and the vision context type all
    # present, the model is in the vision drive -- the state the S2 gate
    # held out until these landed together.
    assert issubclass(MistralLarge4Model, SupportsVisionEncoding)


def test_the_context_stores_its_own_scatter_indices() -> None:
    """The stored indices shadow the span-derived property.

    An image span holds break and end tokens, which embed as text, so the
    scatter indices are the image tokens' own positions -- not every
    position of the span.
    """
    context = MistralLarge4VisionContext.new_padding_context(
        max_length=16, model_name="m"
    )
    assert context.image_token_indices.shape == (0,)
    context.image_token_indices = np.array([3, 7], dtype=np.int32)
    np.testing.assert_array_equal(
        context.image_token_indices, np.array([3, 7], dtype=np.int32)
    )


def test_the_processor_grows_each_placeholder_into_rows() -> None:
    """Rows of image tokens, break separators, one end token; pixels [3,H,W].

    A 56x56 image is two token rows of two image tokens: one break token
    after the first row, the end token where the last row's break would be.
    """
    from PIL import Image

    processor = _processor()
    image = Image.new("RGB", (56, 56))
    processed = processor(text="[IMG]", images=[image])
    ids = processed["input_ids"][0]
    assert ids == [10, 10, 12, 10, 10, 13]
    pixels = processed["pixel_values"][0][0]
    assert pixels.shape == (3, 56, 56)
    assert pixels.dtype == np.float32
    # Two placeholders take two images, in order: each 56x56 image grows
    # into two rows of two image tokens.
    two = processor(text="[IMG] and [IMG]", images=[image, image])
    assert two["input_ids"][0].count(10) == 8
    assert two["input_ids"][0].count(13) == 2
    with pytest.raises(ValueError):
        processor(text="[IMG] and [IMG]", images=[image])


@pytest.mark.parametrize(
    ("original", "resized"),
    [
        ((20, 20), (28, 28)),
        ((500, 333), (504, 336)),
        ((1536, 2048), (1176, 1540)),
        ((1540, 1540), (1540, 1540)),
        ((56, 56), (56, 56)),
        ((57, 57), (84, 84)),
    ],
    ids=["c1", "c2", "c3", "c4", "c5", "c6"],
)
def test_the_processor_resizes_like_the_reference(
    original: tuple[int, int], resized: tuple[int, int]
) -> None:
    """The S1 golden sizing table: shrink past 1540 with floor, round every
    side up to a 28-pixel token, never upscale."""
    assert _processor()._resized_shape(*original) == resized


def _golden_cases() -> list[dict[str, Any]]:
    manifest_path = GOLDEN / "manifest.json"
    if not manifest_path.is_file():
        # Without the fixture there is no case to parametrize with; an
        # empty set skips these tests instead of failing at collection.
        return []
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if isinstance(cases, dict):
        cases = list(cases.values())
    assert cases, f"{GOLDEN} holds no golden case"
    return cases


def _golden_array(case_id: str, stem: str, key: str) -> np.ndarray:
    path = GOLDEN / f"{case_id}.{stem}.safetensors"
    with safe_open(str(path), "numpy") as handle:
        return np.asarray(handle.get_tensor(key))


def _fake_pipeline_config() -> Any:
    """The config reads a tokenizer performs, over a text-only pipeline."""
    return SimpleNamespace(
        model=SimpleNamespace(
            kv_cache=SimpleNamespace(enable_prefix_caching=False),
            huggingface_config=PretrainedConfig(
                eos_token_id=None,
                text_config=PretrainedConfig(vocab_size=204800),
            ),
            generation_config=SimpleNamespace(eos_token_id=None),
        ),
        draft_model=None,
    )


def _image_request(case: dict[str, Any]) -> Any:
    """A chat request carrying the case's golden PNG images."""
    from max.pipelines.modeling.types import TextGenerationRequest
    from PIL import Image

    paths = [
        GOLDEN.parent / "images" / f"{name}.png" for name in case["images"]
    ]
    images = [Image.open(path) for path in paths]
    content: list[dict[str, str]] = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": case["text"]})
    conversation = [{"role": "user", "content": content}]
    return TextGenerationRequest(
        request_id="req-" + case["id"],
        model_name="m",
        messages=cast(Any, conversation),
        # The base tokenizer gates on `images` holding the encoded bytes;
        # the pre-decoded PIL objects ride beside them.
        images=[path.read_bytes() for path in paths],
        decoded_images=cast(Any, images),
    )


@pytest.fixture(scope="module")
def fixture_tokenizer() -> MistralLarge4VisionTokenizer | None:
    """The tokenizer over the fixture checkpoint's own files, when present."""
    if not (CKPT / "tokenizer.json").is_file() or not GOLDEN.is_dir():
        return None
    return MistralLarge4VisionTokenizer(str(CKPT), _fake_pipeline_config())


@pytest.mark.skipif(
    not (CKPT / "tokenizer.json").is_file() or not GOLDEN.is_dir(),
    reason="the fixture checkpoint or the S1 goldens are not on this box",
)
@pytest.mark.parametrize(
    "case",
    _golden_cases(),
    ids=lambda case: case["id"] if isinstance(case, dict) else "none",
)
def test_the_token_layout_matches_the_s1_goldens(
    case: dict[str, Any], fixture_tokenizer: MistralLarge4VisionTokenizer
) -> None:
    """The whole tokenizer path, exact against the reference record.

    The chat template renders the case's messages, the processor grows the
    placeholders, and the context records the image spans and the image
    tokens' own positions -- every one of which the S1 golden run recorded.
    """
    tokenizer = fixture_tokenizer
    context = asyncio.run(tokenizer.new_context(_image_request(case)))

    golden_ids = _golden_array(case["id"], "inputs", "input_ids")
    tokens = np.asarray(context.tokens.all)
    assert tokens.shape == golden_ids.shape, "token count"
    np.testing.assert_array_equal(tokens, golden_ids)

    pos_image = _golden_array(case["id"], "tokens", "pos_image")
    np.testing.assert_array_equal(
        np.asarray(context.image_token_indices, dtype=pos_image.dtype),
        pos_image,
    )

    sizes = _golden_array(case["id"], "inputs", "image_sizes")
    rows = _golden_array(case["id"], "embeddings", "per_image_rows")
    assert len(context.images) == len(sizes)
    total_rows = 0
    for image, (height, width), count in zip(
        context.images, sizes, rows, strict=True
    ):
        assert image.pixel_values.shape == (3, int(height), int(width))
        assert image.num_embedding_rows == int(count)
        assert image.image_hash is not None
        span = tokens[image.start_idx : image.end_idx]
        # The span holds the image tokens plus one separator per row but
        # the last; the end token sits outside, after the span.
        assert (span == 10).sum() == int(count)
        assert (span == 12).sum() == int(height) // 28 - 1
        assert tokens[image.end_idx] == 13
        total_rows += int(count)
    assert total_rows == int(pos_image.shape[0])
    assert context.needs_vision_encoding


@pytest.mark.skipif(
    not GOLDEN.is_dir(),
    reason="the S1 goldens are not on this box",
)
@pytest.mark.parametrize(
    "case",
    _golden_cases(),
    ids=lambda case: case["id"] if isinstance(case, dict) else "none",
)
def test_the_processor_pixels_match_the_goldens(case: dict[str, Any]) -> None:
    """The pixel path measured against the S1 golden pixel arrays.

    An image that keeps its size agrees with the recorded pixels to
    float32 rounding (the reference normalize rounds differently by at
    most 2 ulp, whichever order the two paths round in); a resized image
    carries the PIL-versus-torchvision bicubic delta, bounded far under
    what a wrong mean, std, rescale or channel order would show. A ragged
    golden batch records a padded rectangle; the image owns the leading
    window of it.
    """
    from PIL import Image

    processor = _processor()
    golden_px = _golden_array(case["id"], "inputs", "pixel_values")
    sizes = _golden_array(case["id"], "inputs", "image_sizes")
    per_golden = (
        [golden_px[i] for i in range(golden_px.shape[0])]
        if golden_px.ndim == 4
        else [golden_px]
    )
    paths = [
        GOLDEN.parent / "images" / f"{name}.png" for name in case["images"]
    ]
    for path, size, golden in zip(paths, sizes, per_golden, strict=True):
        image = Image.open(path)
        ours = processor._pixels(image)
        height, width = int(size[0]), int(size[1])
        assert ours.shape == (3, height, width), case["id"]
        window = golden[:, :height, :width]
        max_diff = float(
            np.abs(ours.astype(np.float64) - window.astype(np.float64)).max()
        )
        if processor._resized_shape(image.height, image.width) == (
            image.height,
            image.width,
        ):
            np.testing.assert_allclose(ours, window, rtol=0, atol=1e-6)
        else:
            print(
                f"{case['id']}: resized-image pixel maxdiff {max_diff:.2e}",
                flush=True,
            )
            assert max_diff <= 0.05, case["id"]


def _params() -> dict[str, Any]:
    """The checkpoint's own geometry, cut down to two text layers."""
    params = copy.deepcopy(
        json.loads((_TESTDATA / "ml4_nvfp4_params.json").read_text())
    )
    params["n_layers"] = 2
    params["nope"] = "rope,rope"
    dsa = params["dsa"]
    dsa["sharing"] = "full,full"
    kv_sharing = params["kv_sharing"]
    kv_sharing["source_layers_str"] = "0,1"
    moe = params["moe"]
    moe["first_k_dense_replace"] = 1
    return params


def _seeded_model(
    n_devices: int = 1,
) -> tuple[MistralLarge4Model, MistralLarge4Config]:
    """A MistralLarge4Model instance far enough built to run graph_specs.

    The instance is seeded by hand: graph_specs reads the parsed checkpoint,
    the config, the batch processor and the runtime switches, all of which a
    real construction would hold by the time it compiles. The checkpoint
    values are the two lazily built modules' own parameters -- zero
    allocation, the production names.
    """
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    device_refs = [DeviceRef.GPU(i) for i in range(n_devices)]
    full_kind = "full"
    shared_kind = "shared"
    kv_params = build_kv_params(
        rope_layers=list(normalized.layers.rope),
        kv_source_layers=list(normalized.layers.kv_source),
        indexer_types=[
            full_kind if owns else shared_kind
            for owns in normalized.layers.owns_indexer
        ],
        num_attention_heads=normalized.n_heads,
        index_head_dim=normalized.index_head_dim,
        index_topk=normalized.index_topk,
        devices=device_refs,
        kv_cache_config=KVCacheConfig(kv_cache_page_size=128),
        cache_dtype=DType.float8_e4m3fn,
    )
    config = MistralLarge4Config.from_normalized(
        normalized,
        kv_params=kv_params,
        devices=device_refs,
        max_seq_len=256,
        graph_mode="prefill",
    )
    config.quant_config = config.nvfp4_config
    config.mesh = DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)), (n_devices,), ("tp",)
    )
    config.encodes_images = True
    # The module harvest below builds the two-layer text module at the
    # mesh's own width, and a multi-device build refuses to place the 128
    # routed experts without splitting them: the EP config the runtime init
    # would build is seeded here first, so the harvest and the spec build
    # see the same split.
    if n_devices > 1:
        config.ep_config = config.expert_parallel_config(
            ModuleRuntime(
                mesh=config.mesh,
                data_parallel_degree=1,
                max_batch_input_tokens=512,
                max_batch_total_tokens=None,
                ep_size=n_devices,
                ep_use_allreduce=False,
            )
        )

    # The hand-seeded instance assigns private attributes a real
    # construction owns, so the local stays Any for the seeding block.
    model = cast(Any, object.__new__(MistralLarge4Model))
    model.devices = [Accelerator(i) for i in range(n_devices)]
    model.device_refs = device_refs
    hf_config = PretrainedConfig(
        text_config=PretrainedConfig(
            hidden_size=6144, rms_norm_eps=1e-5, vocab_size=204800
        ),
        vision_config=PretrainedConfig(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=30,
            num_attention_heads=16,
            num_channels=3,
            patch_size=14,
            image_size=1540,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
        ),
        spatial_merge_size=2,
    )
    # Two devices run the checkpoint the way the branch serves it at TP2:
    # attention tensor-parallel, the routed experts split by EP2. The
    # distributed-runtime init graph_specs runs builds the EP config and
    # the batch manager off these switches, in virtual-device mode without
    # a live NVSHMEM runtime.
    model.pipeline_config = SimpleNamespace(
        runtime=SimpleNamespace(
            ep_size=1 if n_devices == 1 else n_devices,
            ep_use_allreduce=False,
            ep_fuse_ffn_combine_send=False,
            max_batch_input_tokens=512,
        ),
        model=SimpleNamespace(
            data_parallel_degree=1,
            kv_cache=SimpleNamespace(kv_cache_page_size=128),
            huggingface_config=hf_config,
            quantization_encoding="bfloat16",
            # The resolved-cast shortcut: the encoding is settled, so the
            # resolver never reads the weight repo the stub does not
            # carry.
            _resolved_dtype_cast=(None, None),
        ),
    )
    model.kv_params = kv_params
    ep_config = getattr(config, "ep_config", None)
    model._ep_batch_manager = (
        EPBatchManager(ep_config) if ep_config is not None else None
    )
    # graph_specs sets this through the distributed-runtime init; the
    # ABI-boundary test reads the input types directly.
    model._modulev3_extra_input_types = []
    model.return_logits = ReturnLogits.LAST_TOKEN
    model.return_hidden_states = ReturnHiddenStates.NONE
    # What the real construction configures for the config-creation path
    # the language-only MEF test drives: the model reads both off its
    # memory plan.
    model.memory_plan = SimpleNamespace(
        planned_max_length=256, planned_max_batch_total_tokens=None
    )
    model._lora_manager = None
    model._batch_processor = DeepseekV3ModuleV3BatchProcessor(
        config,
        BatchProcessorRuntime(
            pipeline_config=model.pipeline_config,
            devices=model.devices,
            return_logits=ReturnLogits.LAST_TOKEN,
            max_batch_size=1,
            max_seq_len=256,
            max_batch_input_tokens=512,
        ),
    )
    with (
        F.lazy(),
        default_device(config.mesh),
        default_dtype(DType.bfloat16),
    ):
        text_module = MistralLarge4(
            config, kv_params, cast(Any, model)._ep_batch_manager
        )
        vision_module = MistralLarge4Vision(VISION_CONFIG)
    # The modules' own lazy weights stand in for the parsed checkpoint: the
    # production names, no allocation, exactly what a ModuleV3 graph stages
    # with when it is handed no weights of its own.
    model._checkpoint = {
        **dict(cast(Any, text_module).parameters),
        **dict(cast(Any, vision_module).parameters),
    }
    return cast(MistralLarge4Model, model), config


def test_graph_specs_traces_both_graph_roles() -> None:
    """The decoder carries the vision tail; the tower sits on device 0.

    Staged on a virtual B200: graph staging only, no compile, no GPU. The
    seeded checkpoint's vision names are the vision module's own
    parameters, so both weight slices carry the production names without an
    allocation.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        model, config = _seeded_model()
        specs = model.graph_specs(config)
        assert set(specs) == {GraphRole.DECODER, GraphRole.VISION}

        vision = specs[GraphRole.VISION]
        assert vision.name == "MistralLarge4Vision"
        # The spec passes the tower's own ABI through, so compare the two
        # rather than restating its arity here.
        assert list(vision.input_types) == vision_input_types(
            VISION_CONFIG, DeviceRef.GPU(0)
        )
        vision_weights = set(vision.weights or {})
        assert vision_weights == {
            name
            for name in model._checkpoint
            if name.startswith(VISION_PREFIXES)
        }
        patch_conv = "vision_encoder.patch_conv"
        assert patch_conv in vision_weights
        first_wq = "vision_encoder.transformer.layers.0.attention.wq.weight"
        assert first_wq in vision_weights

        decoder = specs[GraphRole.DECODER]
        decoder_weights = set(decoder.weights or {})
        assert not any(
            name.startswith(VISION_PREFIXES) for name in decoder_weights
        )
        assert decoder_weights | vision_weights == set(model._checkpoint)

        input_types = list(decoder.input_types)
        embeddings_type, indices_type = input_types[-2:]
        assert isinstance(embeddings_type, TensorType)
        assert embeddings_type.dtype == DType.bfloat16
        assert list(embeddings_type.shape) == [
            "num_image_tokens",
            config.hidden_size,
        ]
        assert embeddings_type.device == DeviceRef.GPU(0)
        assert isinstance(indices_type, TensorType)
        assert indices_type.dtype == DType.int32
        assert list(indices_type.shape) == ["total_image_tokens"]
        assert indices_type.device == DeviceRef.GPU(0)

        decoder_trace = str(decoder.trace())
        # The scatter: image rows into the prompt's placeholder positions,
        # once, fed by the two vision inputs at the ABI tail.
        assert (
            decoder_trace.count("gather_scatter.scatter_nd.skip_oob_indices")
            == 1
        )

        vision_trace = str(vision.trace())
        assert vision_trace.count("attention.mha.ragged.no_cache") == (
            VISION_CONFIG.vision_num_hidden_layers
        )
        assert patch_conv in vision_trace
        self_attn = ".self_attn."
        assert self_attn not in vision_trace
    finally:
        set_virtual_device_count(0)


def test_a_module_without_the_vision_flag_keeps_the_text_abi() -> None:
    """``encodes_images`` is the ABI boundary the fused graph relies on.

    The DSpark pairing builds its target with the flag off and traces
    ``verify`` against the text-only input list; the pipeline model turns
    the flag on and its decoder declares the vision tail.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        model, config = _seeded_model()
        with_images = model._get_compile_input_types(config)
        config.encodes_images = False
        without_images = model._get_compile_input_types(config)
        config.encodes_images = True
        assert len(without_images) == len(with_images) - 2
        assert without_images == tuple(with_images[:-2])
    finally:
        set_virtual_device_count(0)


def test_a_text_only_batch_never_enters_vision_encoding() -> None:
    """The pipeline's vision drive skips a batch with no image to encode.

    A text-only request's context holds no image, so the run returns before
    the model's packing or encoding is ever reached -- text-only traffic
    never pays the tower. The spies hold that to the methods themselves:
    had the drive reached the model, the pack or the encode would have run.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        model, _ = _seeded_model()
        context = MistralLarge4VisionContext.new_padding_context(
            max_length=16, model_name="m"
        )
        assert not context.needs_vision_encoding

        def _record(name: str) -> Any:
            def _spy(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                raise AssertionError(
                    f"a text-only batch reached the vision drive: {name}"
                )

            return _spy

        seeded: Any = model
        seeded.pack_vision_inputs = _record("the vision pack")
        seeded.vision_execute = _record("the vision encode")
        cache: VisionEncoderCache[MistralLarge4VisionContext] = (
            VisionEncoderCache()
        )
        result = cache.run_vision_encode(
            cast(Any, model), [[context]], model.devices
        )
        assert result is None
    finally:
        set_virtual_device_count(0)


def _vision_hf_config() -> Any:
    """The multimodal HF config shape the vision-cache facts read."""
    return PretrainedConfig(
        text_config=PretrainedConfig(hidden_size=6144),
        vision_config=PretrainedConfig(patch_size=14, image_size=1540),
        spatial_merge_size=2,
    )


def test_ml4_reserves_a_vision_encoder_cache() -> None:
    """Both vision-cache facts published, so the cache runs enabled.

    ``arch_has_vision_tower`` needs the per-entry estimate beside the row
    spec; without the estimate the pipeline would serve with the cache
    silently disabled -- every chunked-prefill chunk of an image prompt
    re-running the tower, every request re-encoding its images. The worst
    case is the largest image the processor admits: 1540 pixels on a side
    is 110 patches of 14, 55x55 merged rows, one BF16 row each at the text
    hidden width.
    """
    hf_config = _vision_hf_config()
    tokens_per_side = 1540 // (14 * 2)
    entry = MistralLarge4Config.estimate_vision_cache_entry_bytes(hf_config)
    assert entry == tokens_per_side**2 * 6144 * 2
    assert entry > 0
    assert arch_has_vision_tower(MistralLarge4Config, hf_config)

    # The stubs carry only the fields the planner reads; the calls are the
    # production ones, so the arguments stay Any for the stub block.
    arch: Any = SimpleNamespace(
        name="MistralLarge4ForConditionalGeneration_ModuleV3",
        config=MistralLarge4Config,
    )
    pipeline_config: Any = SimpleNamespace(
        runtime=SimpleNamespace(vision_cache_utilization=0.05),
        model=SimpleNamespace(huggingface_config=hf_config),
    )
    row = _vision_row(pipeline_config, arch)
    reserved, plan = vision_cache_plan(
        pipeline_config,
        row,
        cache_bytes=256 * 1024**2,
        n_devices=1,
    )
    assert reserved > 0
    assert plan is not None
    cache: VisionEncoderCache[Any] = VisionEncoderCache(plan, devices=[CPU()])
    assert cache.enabled


def test_the_shared_inputs_pack_the_text_only_buffers_unchanged() -> None:
    """``DeepseekV3Inputs.buffers`` is the pre-vision layout without the pair.

    The vision fields default empty and only the pipeline's vision seam
    sets them, so a text-only architecture packs exactly the buffers it
    always did -- the shared file's blast radius stops at an unset field.
    With the pair set, the ABI tail appends the device-0 element of each.
    """
    tokens = Buffer.from_numpy(np.zeros(4, np.int64))
    return_n_logits = Buffer.from_numpy(np.array([1], np.int32))
    offsets = Buffer.from_numpy(np.array([0, 4], np.uint32))
    context_length = Buffer.from_numpy(np.array([128], np.int32))
    inputs = DeepseekV3Inputs(
        tokens=tokens,
        input_row_offsets=offsets,
        return_n_logits=return_n_logits,
        batch_context_lengths=[context_length],
    )
    text_only = inputs.buffers
    assert len(text_only) == 4
    for got, want in zip(
        text_only,
        (tokens, return_n_logits, offsets, context_length),
        strict=True,
    ):
        assert got is want

    embeddings = [Buffer.from_numpy(np.zeros((0, 6144), np.float32))]
    indices = [Buffer.from_numpy(np.zeros(0, np.int32))]
    inputs.vision_embeddings = embeddings
    inputs.vision_scatter_indices = indices
    with_pair = inputs.buffers
    assert len(with_pair) == len(text_only) + 2
    for i in range(len(text_only)):
        assert with_pair[i] is text_only[i]
    assert with_pair[-2] is embeddings[0]
    assert with_pair[-1] is indices[0]


def _trace_error(spec: GraphSpec) -> ValueError | None:
    """Traces ``spec``, returning its verifier error instead of raising."""
    try:
        str(spec.trace())
    except ValueError as error:
        return error
    return None


def _no_vision_graph(self: Any, model_config: Any) -> bool:
    """What the MEF exporter patches in for a language-only build."""
    del self, model_config
    return False


_EP_COMBINE_CONCAT_DIM = (
    "add(div(add(total_seq_len, 1), 2), div(total_seq_len, 2))"
)
"""The EP-combine operand the rebind gap trips on.

The EP combine concatenates ceil(s/2) + floor(s/2) EP-split rows, and
the shape verifier reports the concatenated dimension by its symbolic
spelling -- a key no vision-side fault could produce.
"""


def test_empty_vision_embeddings_follow_the_devices_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empties answer the device list the cache asks for.

    The pipeline asks with the model devices, whose empties are built once
    and reused; a caller handing a different list gets fresh zero-row
    buffers for it, not the cached ones.

    The allocation is stubbed: a virtual device accepts no memory operation,
    so a real zero-row buffer cannot land on one, and what this gate is about
    is the device list the wiring forwards and the geometry it sizes to -- not
    the copy.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(2)
    allocated: list[tuple[list[Any], int, DType]] = []

    def record(
        devices: list[Any], hidden_size: int, dtype: DType = DType.bfloat16
    ) -> list[Any]:
        allocated.append((list(devices), hidden_size, dtype))
        return [object() for _ in devices]

    monkeypatch.setattr(ml4_model, "create_empty_image_embeddings", record)
    try:
        model, _ = _seeded_model()
        asked = model.empty_vision_embeddings(model.devices)
        assert asked is model._empty_vision_embeddings
        assert allocated == [
            (
                model.devices,
                model._vision_config.hidden_size,
                model._vision_config.dtype,
            )
        ]

        others: list[Device] = [Accelerator(0), Accelerator(1)]
        fresh = model.empty_vision_embeddings(others)
        assert fresh is not model._empty_vision_embeddings
        assert len(fresh) == 2
        assert allocated[-1][0] == others
        assert allocated[-1][1] == model._vision_config.hidden_size
        assert allocated[-1][2] == model._vision_config.dtype
    finally:
        set_virtual_device_count(0)


def test_graph_specs_stages_the_tp_pair_on_a_two_device_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two devices: the vision pair replicates onto every rank of the mesh.

    The tower still runs once, on device 0; the language graph takes the
    rows and the indices from device 0 and broadcasts them so every
    tensor-parallel rank scatters the same rows at the same replicated
    positions. Staged on a virtual pair of B200s -- the multi-device
    build needs the routed experts under EP2 -- with the decoder trace
    held behind a known rebind gap: the EP combine concatenates
    ceil(s/2) + floor(s/2) EP-split rows, which the graph verifier
    cannot prove equal to s without an explicit rebind. The gap is
    proven vision-independent by a control -- the same two-device
    staging with the vision graph excluded, exactly what a language-only
    MEF compiles, fails the same verifier at the same EP-combine
    operand -- so the xfail keys on that operand rather than on the
    generic op-constructor wrapper a vision fault would also trip. The
    served-pipeline TP arm (S5) needs the rebind fix.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(2)
    try:
        model, config = _seeded_model(n_devices=2)
        specs = model.graph_specs(config)
        assert set(specs) == {GraphRole.DECODER, GraphRole.VISION}

        vision = specs[GraphRole.VISION]
        # The tower runs on device 0 alone: every device input sits on
        # GPU(0), beside the one host scalar its ABI carries.
        for t in vision.input_types:
            device = cast(Any, t).device
            assert device == DeviceRef.GPU(0) or device == DeviceRef.CPU()

        decoder = specs[GraphRole.DECODER]
        input_types = list(decoder.input_types)
        embeddings_type, indices_type = input_types[-2:]
        assert isinstance(embeddings_type, TensorType)
        assert embeddings_type.device == DeviceRef.GPU(0)
        assert isinstance(indices_type, TensorType)
        assert indices_type.device == DeviceRef.GPU(0)

        # The control: the same mesh and EP split, the language graph
        # alone. Whatever its trace does, it does without any vision
        # code in the graph.
        monkeypatch.setattr(
            MistralLarge4Model, "_include_vision_graph", _no_vision_graph
        )
        config.encodes_images = False
        control_specs = model.graph_specs(config)
        assert set(control_specs) == {GraphRole.DECODER}
        control_error = _trace_error(control_specs[GraphRole.DECODER])
        config.encodes_images = True

        try:
            decoder_trace = str(decoder.trace())
        except ValueError as error:
            # Only the EP-combine rebind gap may hide this trace, and only
            # when the vision-free control failed at the same operand: a
            # fault in the staged vision pair fails the test.
            if control_error is None or not all(
                _EP_COMBINE_CONCAT_DIM in str(each)
                for each in (error, control_error)
            ):
                raise
            pytest.xfail(
                "the EP combine rows need an explicit rebind at TP>1,"
                " vision or not -- the vision-free control fails the same"
                f" shape verifier: {error}"
            )
        # The staged batch and the vision pair replicate onto the mesh:
        # tokens, offsets, embeddings and indices each transfer.
        assert decoder_trace.count("mo.transfer") >= 4
        assert (
            decoder_trace.count("gather_scatter.scatter_nd.skip_oob_indices")
            >= 1
        )
    finally:
        set_virtual_device_count(0)


def test_the_language_only_mef_build_keeps_the_text_only_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the gen_mef language-only export builds: the decoder alone.

    The exporter replaces the ``_include_vision_graph`` hook for the
    language build, and the production ``_create_model_config`` is what
    turns that hook into the config flag and the config flag into the
    ABI -- so the export drops the vision role and the decoder
    vision-merge inputs together, the text-only language ABI the engine
    ML4 slot map declares until S4 lands the engine-side vision
    contract. The real config-creation path runs on the real class; only
    the config constructor stands in, because the hand-seeded model
    cannot fill the HF config it reads.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        model, config = _seeded_model()

        def _seeded_initialize(pipeline_config: Any, **_: Any) -> Any:
            del pipeline_config
            return config

        monkeypatch.setattr(
            MistralLarge4Model, "_include_vision_graph", _no_vision_graph
        )
        monkeypatch.setattr(
            MistralLarge4Config, "initialize", _seeded_initialize
        )
        language_config = model._create_model_config({})
        assert isinstance(language_config, MistralLarge4Config)
        assert not language_config.encodes_images
        specs = model.graph_specs(language_config)
        assert GraphRole.VISION not in specs
        language_only = model._get_compile_input_types(language_config)

        language_config.encodes_images = True
        with_vision = model._get_compile_input_types(language_config)
        assert len(language_only) == len(with_vision) - 2
        assert language_only == tuple(with_vision[:-2])
    finally:
        set_virtual_device_count(0)


def test_a_towerless_checkpoint_is_refused_while_it_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint without a vision tower never starts serving.

    The architecture has no text-only mode to fall back to: its
    tokenizer expands image placeholders and every prepared batch runs
    the empty-image drive, both of which read the vision geometry. The
    refusal lands in ``_create_model_config``, which the pipeline runs
    while it loads -- a tower-less checkpoint fails with a clear error
    instead of starting and dying on its first request.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    try:
        model, _ = _seeded_model()
        towerless = PretrainedConfig(
            text_config=PretrainedConfig(hidden_size=6144, vocab_size=204800)
        )
        assert getattr(towerless, "vision_config", None) is None
        monkeypatch.setattr(
            model.pipeline_config.model, "huggingface_config", towerless
        )
        with pytest.raises(ValueError, match="vision_config"):
            model._create_model_config({})
    finally:
        set_virtual_device_count(0)
