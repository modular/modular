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
"""Checks how the Qwen3.5 ModuleV3 build registers, without compiling."""

from types import SimpleNamespace
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.outputs import ReturnLogits
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.mtp_qwen3_5_modulev3 import (
    Qwen3_5MTP,
    mtp_qwen3_5_modulev3_draft_arch,
)
from max.pipelines.architectures.qwen3_5_modulev3 import (
    Qwen3_5Decoder,
    qwen3_5_modulev3_arch,
    qwen3_5_text_modulev3_arch,
)
from max.pipelines.architectures.qwen3_5_modulev3.batch_processor import (
    Qwen3_5BatchProcessor,
    context_position_rows,
)
from max.pipelines.architectures.qwen3_5_modulev3.context import (
    Qwen3_5TextAndVisionContext,
)
from max.pipelines.architectures.qwen3_5_modulev3.model_config import (
    Qwen3_5Config,
)
from max.pipelines.architectures.qwen3_5_modulev3.vision import (
    Qwen3_5VisionEncoder,
)
from max.pipelines.context import ImageMetadata, TextContext, TokenBuffer
from max.pipelines.context.context_validators import validate_no_images
from max.pipelines.lib import GraphRole
from max.pipelines.lib.arch_lookup import find_architecture
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.module_registration import (
    ModuleV3DecoderPipelineModel,
    token_position_type,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeModuleV3VisionCachePipelineModel,
    spec_decode_architecture,
)
from max.pipelines.modeling.types import InputModality, RequestID
from max.pipelines.speculative import SpeculativeMethod, spec_adapters


@pytest.fixture(autouse=True, scope="module")
def _registered() -> None:
    register_all_models()


def test_the_flag_selects_the_module_v3_build() -> None:
    for arch in (qwen3_5_modulev3_arch, qwen3_5_text_modulev3_arch):
        name = arch.name.removesuffix("_ModuleV3")
        found = find_architecture(name, prefer_module_v3=True)
        assert found is not None and found.name == arch.name
        legacy = find_architecture(name, prefer_module_v3=False)
        assert legacy is not None and legacy.name == name


def test_the_decoder_and_tower_register_by_role() -> None:
    model = qwen3_5_modulev3_arch.pipeline_model
    assert model is not None
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    assert model.decoder_cls is Qwen3_5Decoder
    assert model.vision_cls is Qwen3_5VisionEncoder
    assert getattr(model, "checkpoints_recurrent_state", False)


def test_spec_decoding_calls_both_module_v3_variants() -> None:
    for arch in (qwen3_5_modulev3_arch, qwen3_5_text_modulev3_arch):
        assert spec_adapters.has_adapter(arch.name)


def test_the_mtp_head_registers_as_a_draft() -> None:
    arch = mtp_qwen3_5_modulev3_draft_arch
    assert arch.modules[GraphRole.DRAFT] is Qwen3_5MTP
    assert arch.draft is not None and arch.draft.method == "mtp"
    assert (
        find_architecture("MTPQwen3_5DraftModel", prefer_module_v3=True) is arch
    )


def test_m_rope_positions_are_declared_only_with_a_tower() -> None:
    device = DeviceRef.GPU()
    declared = token_position_type(
        Qwen3_5Decoder, SimpleNamespace(mrope_enabled=True), device
    )
    assert isinstance(declared, TensorType)
    assert declared.dtype == DType.int64
    assert [str(d) for d in declared.shape] == ["3", "total_seq_len"]
    assert (
        token_position_type(
            Qwen3_5Decoder, SimpleNamespace(mrope_enabled=False), device
        )
        is None
    )


def test_text_positions_count_on_from_the_processed_length() -> None:
    ctx = TextContext(
        request_id=RequestID(),
        max_length=16,
        tokens=TokenBuffer(np.arange(8, dtype=np.int64)),
    )
    ctx.tokens.skip_processing(5)
    rows = context_position_rows(ctx)
    np.testing.assert_array_equal(rows, np.tile(np.arange(5, 8), (3, 1)))


def test_a_verify_window_continues_positions_past_an_image() -> None:
    """A speculative step's drafts take the positions after its token.

    The verify window is longer than the step's active tokens, and past an
    image every one of its positions carries the request's rope delta.
    """
    ctx = Qwen3_5TextAndVisionContext(
        request_id=RequestID(),
        max_length=16,
        tokens=TokenBuffer(np.arange(8, dtype=np.int64)),
        vision_token_ids=[],
        spatial_merge_size=2,
        rope_delta=-3,
        image_token_id=0,
        video_token_id=0,
        vision_start_token_id=0,
        vision_end_token_id=0,
        image_token_indices=np.empty(0, dtype=np.int32),
        decoder_position_ids=np.empty((3, 0), dtype=np.int64),
        vision_data=None,
    )
    ctx.tokens.skip_processing(7)
    rows = context_position_rows(ctx, length=4)
    np.testing.assert_array_equal(rows, np.tile(np.arange(4, 8), (3, 1)))


@pytest.mark.parametrize(
    ("method", "draft"), [("mtp", None), ("dflash2", "DFlash2DraftModel")]
)
def test_every_speculative_pairing_encodes_images(
    method: SpeculativeMethod, draft: str | None
) -> None:
    """A vision checkpoint's pairing encodes its images through the cache.

    Its fused graph takes the image rows and the verify window's M-RoPE
    positions under every method.
    """
    target = qwen3_5_modulev3_arch.name.removesuffix("_ModuleV3")
    pairing = spec_decode_architecture(
        target, method, draft, prefer_module_v3=True
    )
    assert pairing is not None
    assert pairing.pipeline_model is (
        _SpecDecodeModuleV3VisionCachePipelineModel
    )
    assert validate_no_images not in pairing.context_validators
    assert InputModality.IMAGE in pairing.input_modalities
    config = SimpleNamespace(mrope_enabled=True, devices=[DeviceRef.GPU()])
    (positions,) = spec_adapters.trailing_input_types(
        qwen3_5_modulev3_arch.name, method, config
    )
    assert isinstance(positions, TensorType)
    assert [str(d) for d in positions.shape] == ["3", "merged_total_seq_len"]


def _image_prompt() -> Qwen3_5TextAndVisionContext:
    """A prompt whose one image the vision tower has yet to encode."""
    image = 9
    return Qwen3_5TextAndVisionContext(
        request_id=RequestID(),
        max_length=16,
        tokens=TokenBuffer(np.array([1, image, image, 2], dtype=np.int64)),
        vision_token_ids=[image],
        images=[
            ImageMetadata(
                start_idx=1, end_idx=3, pixel_values=np.zeros(1, np.float32)
            )
        ],
        spatial_merge_size=2,
        rope_delta=0,
        image_token_id=image,
        video_token_id=0,
        vision_start_token_id=0,
        vision_end_token_id=0,
        image_token_indices=np.array([1, 2], dtype=np.int32),
        decoder_position_ids=np.tile(np.arange(4, dtype=np.int64), (3, 1)),
        vision_data=None,
    )


def _unbound_batching(mrope_enabled: bool) -> Qwen3_5BatchProcessor:
    """The batching as a speculative pairing builds it: no input tree bound."""
    return Qwen3_5BatchProcessor(
        NonCallableMock(spec=Qwen3_5Config, mrope_enabled=mrope_enabled),
        BatchProcessorRuntime(
            pipeline_config=NonCallableMock(),
            devices=[CPU()],
            return_logits=ReturnLogits.VARIABLE,
            max_batch_size=1,
            max_seq_len=16,
            max_batch_input_tokens=16,
        ),
    )


def test_a_pairings_target_batching_takes_image_prompts() -> None:
    """The fused graph passes M-RoPE positions as its own trailing input.

    So the target's batching binds no input tree, and still batches an image
    prompt for a checkpoint that runs M-RoPE.
    """
    batching = _unbound_batching(mrope_enabled=True)
    prompt = _image_prompt()
    assert batching.input_tree is None and prompt.needs_vision_encoding
    inputs = batching.prepare_initial_token_inputs(
        [[prompt]], kv_cache_inputs=NonCallableMock(spec=KVCacheInputsPerDevice)
    )
    np.testing.assert_array_equal(inputs.tokens.to_numpy(), [1, 9, 9, 2])
    assert inputs.token_positions is None


def test_a_checkpoint_without_m_rope_refuses_image_prompts() -> None:
    batching = _unbound_batching(mrope_enabled=False)
    with pytest.raises(ValueError, match="cannot serve image prompts"):
        batching.prepare_initial_token_inputs(
            [[_image_prompt()]],
            kv_cache_inputs=NonCallableMock(spec=KVCacheInputsPerDevice),
        )
