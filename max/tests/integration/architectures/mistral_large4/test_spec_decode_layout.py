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
"""The fused Mistral Large 4 DSpark graph's input layout, without a GPU.

Mach's block-spec executor binds a spec MEF's inputs *positionally*, against a
layout it pins in Rust (``block_spec_step_executor.rs``'s
``the_mistral_large4_layout_gives_every_group_an_mla_record``): 419 slots at
TP8/EP8/DP1, the four KV records starting at 3, 91, 179 and 291, record widths
of 11, 11, 14 and 11, the indexer's record folding the scales group at +6..+8,
the target's 22-slot EP tail at 379-400, and the signal buffers trailing at
411-418. Nothing in MAX's runtime would notice a drift -- a renamed or dropped
slot moves every input after it and the engine fails the boot or, worse, feeds
the graph transposed inputs -- so the layout is pinned here against those
numbers.

The signature is built through the same objects the pipeline builds: the
target's KV tree from the real ``config.json``, the pairing's ``_kv_tree`` (which
adds the drafter's own leaf), the ``BlockDriver`` that declares the canonical
spec signature, and the target adapter's declarations, whose EP config is the
production builder's. No module is traced and no device is touched: what a
trace adds on top is the mesh's signal buffers, which the adapter's
``signal_buffer_index`` places.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.legacy.comm import Signals
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.nn.legacy.comm.ep.ep_manager import get_ep_local_sync_counters_size
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.architectures.dspark_mistral_large4_modulev3.arch import (
    dspark_mistral_large4_modulev3_draft_arch,
)
from max.pipelines.architectures.mistral_large4_modulev3.arch import (
    mistral_large4_modulev3_arch,
)
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    INDEXER_GROUP,
    NOPE_MLA_GROUP,
    ROPE_MLA_GROUP,
    MistralLarge4Config,
)
from max.pipelines.architectures.mistral_large4_modulev3.context import (
    MistralLarge4VisionContext,
)
from max.pipelines.context.context_validators import validate_no_images
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.arch_lookup import ARCH_LOOKUP
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _input_type_spec,
    _kv_tree,
    _Pairing,
    spec_decode_architecture,
)
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.modeling.types import InputModality
from max.pipelines.speculative import SpeculativeConfig, spec_adapters
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
)
from max.tensor.sharding import DeviceMesh

TESTDATA = Path("max/tests/integration/architectures/mistral_large4/testdata")
"""Where the checkpoint's generated config snapshots live, relative to runfiles."""

DEVICES = 8
"""The deployment's rank count: TP8 attention, one attention replica."""

PAGE_SIZE = 128

RECORD_STARTS = [3, 91, 179, 291]
"""Where each KV record starts, from Mach's layout test."""

RECORD_WIDTHS = [11, 11, 14, 11]
"""Slots one device's record claims: the eight an MLA group carries, the two
host partition scalars, the mixed-step split record every paged leaf on this
export declares last, plus the scales triple the indexer folds."""

EP_START = 379
"""Where the EP tail starts: right after the last KV record.

The deployment is EP8 at one attention replica, so the target's forward reads
the same collective buffers its base graph declares, and the fused graph
declares them after the KV tree.
"""

EP_COUNTERS = 16
"""Atomic counter buffers: two collective groups over eight ranks."""

EP_COUNTER_STATIC_SHAPE = 480
"""Int32 words in each counter buffer, as the EP config sizes them."""

EP_POINTER_TABLES = 6
"""Symmetric-heap pointer tables: three per collective group."""

TAIL_START = 401
"""Where the sampling and bitmask run starts: past the EP tail."""

SIGNAL_START = 411
"""Where the mesh's signal buffers start, which is Mach's Trailing placement."""

TOTAL_SLOTS = 419
"""``3 + 8 * (11 + 11 + 14 + 11) + 22 + 10 + 8``, the layout's arithmetic."""

MIXED_STEP_SPLIT_SLOTS = 7
"""Slots of the batch's mixed-step split record, which every paged leaf
declares last (``KVCacheInputsPerDevice.mixed_step_split``)."""


def _hf_config() -> Any:
    """The checkpoint's generated ``config.json``, as MAX reads it.

    Its nested objects are namespaces too: the generated file keeps the text
    model's fields under ``text_config``, which every reader here reaches by
    attribute.
    """

    def namespace(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(
                **{key: namespace(item) for key, item in value.items()}
            )
        return value

    return namespace(
        json.loads((TESTDATA / "ml4_nvfp4_hf_config.json").read_text())
    )


def _devices() -> list[DeviceRef]:
    return [DeviceRef.GPU(i) for i in range(DEVICES)]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stand-in checkpoint directory holding the real ``params.json``.

    The drafter's taps, depth and window are read from ``params.json``, so the
    snapshot beside the generated ``config.json`` is copied under the name a
    served checkpoint uses.
    """
    directory = tmp_path_factory.mktemp("ml4-checkpoint")
    (directory / "params.json").write_text(
        (TESTDATA / "ml4_nvfp4_params.json").read_text()
    )
    return directory


def _pipeline_config(checkpoint: Path) -> Any:
    """What the pairing reads of a served config that prefers ModuleV3.

    ``checkpoint`` stands in for the served model directory: the drafter's
    geometry comes from the checkpoint's ``params.json``, so the fixture copies
    the testdata snapshot there under that name.
    """
    return cast(
        Any,
        SimpleNamespace(
            models=SimpleNamespace(
                main_architecture_name="MistralLarge4ForConditionalGeneration"
            ),
            # No draft model: the drafter ships inside the target checkpoint,
            # so the target's ``checkpoint_drafts`` names it.
            draft_model=None,
            speculative=SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=3
            ),
            needs_bitmask_constraints=True,
            runtime=SimpleNamespace(
                # EP8 beside TP8 attention: no data-parallel axis, but every
                # rank holds a share of the routed experts.
                reasoning_parser=None,
                prefer_module_v3=True,
                ep_size=DEVICES,
                ep_fuse_ffn_combine_send=False,
                ep_use_allreduce=True,
                max_batch_input_tokens=1024,
            ),
            model=SimpleNamespace(
                data_parallel_degree=1,
                pipeline_stage=None,
                huggingface_config=_hf_config(),
                kv_cache=KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
                model_path=str(checkpoint),
                weight_path=[],
                # The drafter's geometry is read from the checkpoint's
                # ``params.json``, which the tap resolution locates through the
                # weight repo handle -- one a served config always carries, and
                # this stand-in is the local directory it names.
                huggingface_weight_repo=SimpleNamespace(
                    repo_type="local",
                    repo_id=str(checkpoint),
                    revision="main",
                    subfolder=None,
                    local_path=str(checkpoint),
                ),
            ),
        ),
    )


def _pairing() -> _Pairing:
    return _Pairing(
        "MistralLarge4ForConditionalGeneration",
        mistral_large4_modulev3_arch,
        dspark_mistral_large4_modulev3_draft_arch,
    )


def _tree(checkpoint: Path) -> MultiKVCacheParams:
    """The KV tree the fused graph declares, as the pipeline builds it."""
    config = _pipeline_config(checkpoint)
    target = MistralLarge4Config.construct_kv_params(
        config.model.huggingface_config,
        config,
        _devices(),
        config.model.kv_cache,
        DType.float8_e4m3fn,
    )
    return _kv_tree(config, _pairing(), target)


def _slots(leaf: KVCacheParams) -> int:
    """How many graph inputs one device's record of ``leaf`` declares.

    A record's width is the number of its fields the leaf fills in; a
    speculative MLA leaf fills ten, a plain one eight, and a quantized owner
    folds three more for its scales group.
    """
    return len(
        [
            field
            for field in dataclasses.fields(leaf.get_symbolic_inputs()[0])
            if getattr(leaf.get_symbolic_inputs()[0], field.name) is not None
        ]
    )


def _ep_model_config() -> Any:
    """The target's config, as the EP builder reads it.

    The checkpoint's numbers: 128 routed experts four per token on a 6144-wide
    residual, and the mixed NVFP4-routed / FP8-shared quantization whose shared
    expert cannot be stacked with the packed routed ones.
    """
    return SimpleNamespace(
        dtype=DType.bfloat16,
        hidden_size=6144,
        num_experts_per_tok=4,
        n_routed_experts=128,
        n_shared_experts=1,
        quant_config=SimpleNamespace(
            is_nvfp4=True,
            shared_experts_use_quant=lambda dtype: False,
        ),
    )


def _ep_config(
    pipeline_config: Any,
    ep_size: int,
    n_devices: int,
    model_config: Any,
) -> EPConfig:
    """Runs the config's own expert-parallel table on the fixture's stand-in.

    The fused build asks ``MistralLarge4Config.expert_parallel_config``, so
    this reaches the same method through a stand-in config: the tail
    arithmetic below is checked against the production decision table rather
    than against a copy of it.
    """
    return DeepseekV3_2Config.expert_parallel_config(
        model_config,
        ModuleRuntime(
            # The table reads the mesh's device count alone, so it is sized
            # with host devices: this target touches no accelerator.
            mesh=DeviceMesh(
                tuple(CPU() for _ in range(n_devices)),
                (n_devices,),
                ("tp",),
            ),
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            max_batch_input_tokens=(
                pipeline_config.runtime.max_batch_input_tokens
            ),
            max_batch_total_tokens=None,
            ep_size=ep_size,
            ep_use_allreduce=pipeline_config.runtime.ep_use_allreduce,
        ),
    )


def _ep_manager(checkpoint: Path) -> EPBatchManager:
    """The collective manager the deployment's build creates.

    Its config is the production builder's, not a restatement of it: the arm
    ``gen_mef --ep-size 8 --ep-use-allreduce`` exports, every rank a
    participant, 128 routed experts, four per token.
    """
    return EPBatchManager(
        _ep_config(
            _pipeline_config(checkpoint), DEVICES, DEVICES, _ep_model_config()
        )
    )


def _ep_target(checkpoint: Path) -> Any:
    """A target built with an EP manager, which is what names the tail."""
    return SimpleNamespace(ep_batch_manager=_ep_manager(checkpoint))


def _input_spec(
    checkpoint: Path, *, ep: bool = True
) -> SpecDecodeInputTypeSpec:
    """The spec the fused graph's signature is declared from.

    Built through the production helper ``_driver_call`` uses, so a variation
    point it stops passing -- the adapter's EP tail, say -- fails here instead
    of only in a served boot: an undistributed ABI with the target's own EP
    tail and the width the graph declares, the input spec's bitmask triple, and
    no in-thinking slot because Mistral Large 4's adapter does not relax the
    acceptance rule.
    """
    config = _pipeline_config(checkpoint)
    arch = _pairing().adapter_arch
    thinking = (
        config.runtime.reasoning_parser is not None
        and spec_adapters.relaxed_thinking_acceptance(arch)
    )
    target = (
        _ep_target(checkpoint) if ep else SimpleNamespace(ep_batch_manager=None)
    )
    # ``_driver_call`` counts the tail off the built module -- whose manager
    # announces exactly these inputs -- and hands the count to the spec, which
    # is where the batch processor reads it back.
    declared = len(_ep_manager(checkpoint).input_types()) if ep else 0
    spec = _input_type_spec(
        config,
        _devices(),
        arch,
        target,
        distributed=False,
        include_signal_buffers=True,
        trace_owns_signal_buffers=True,
        signal_buffer_index=spec_adapters.signal_buffer_index(arch),
        thinking=thinking,
        vision=False,
        ep_tail_arity=declared,
    )
    assert spec.ep_tail_arity == (declared if ep else 0)
    return spec


def _signature(checkpoint: Path, *, ep: bool = True) -> tuple[Any, ...]:
    """The fused graph's declared input types, in Mach's positional order.

    Args:
        checkpoint: The stand-in checkpoint directory.
        ep: Whether the build runs expert parallelism, which is what declares
            the target's collective tail.
    """
    config = _pipeline_config(checkpoint)
    tree = _tree(checkpoint)
    speculative = config.speculative
    # A proposer's declarations are all the driver's signature reads of it, and
    # its stub verify is never called: only ``input_types`` is asked for.
    proposer = SimpleNamespace(
        block_size=3,
        mask_token_id=0,
        samples_from_anchor=True,
        supports_zero_draft_rows=False,
    )
    manager = _ep_manager(checkpoint) if ep else None
    driver = BlockDriver(
        cast(Any, lambda batch: None),
        cast(Any, proposer),
        target_model=cast(Any, SimpleNamespace()),
        draft_model=cast(Any, SimpleNamespace()),
        input_spec=_input_spec(checkpoint, ep=ep),
        speculative_config=speculative,
        enable_structured_output=config.needs_bitmask_constraints,
        # The target's collectives, on an undistributed signature: this
        # deployment runs EP with one attention replica.
        ep_input_types=() if manager is None else manager.input_types(),
    )
    declared = tuple(driver.input_types(tree))
    # The trace appends the mesh's signal buffers after every declared input
    # (``trace_owns_signal_buffers``), which is Mach's Trailing placement. The
    # exporter records the traced graph, so they are this list's tail.
    signals = Signals(devices=_devices()).input_types()
    assert driver.input_spec.signal_buffer_index is None
    return (*declared, *signals)


def test_the_layout_is_the_one_mach_binds(checkpoint: Path) -> None:
    """419 slots, the four record starts, and the trailing signal buffers.

    These are ``block_spec_step_executor.rs``'s pinned numbers for this model.
    A change here is a Mach change: the engine refuses the boot on arity, or
    silently mis-binds every slot after the one that moved.
    """
    types = _signature(checkpoint)
    assert len(types) == TOTAL_SLOTS
    # The head: tokens, row offsets and the host logit count.
    assert [str(t.dtype) for t in types[:3]] == [
        "DType.int64",
        "DType.uint32",
        "DType.int64",
    ]
    # Each record's first slot is its paged cache blocks, and each record's
    # width is what Mach's layout computes from the group's dispatch schema.
    for start, width in zip(RECORD_STARTS, RECORD_WIDTHS, strict=True):
        assert str(types[start].dtype) in (
            "DType.float8_e4m3fn",
            "DType.bfloat16",
        ), (start, types[start])
        next_start = start + DEVICES * width
        assert next_start in (*RECORD_STARTS, EP_START), (start, next_start)


def test_the_drafters_record_is_bf16_and_eleven_slots_wide(
    checkpoint: Path,
) -> None:
    """The drafter's own group: bf16 latent rows, eleven slots a device.

    A leaf built without its speculative method declares nine, which would put
    every later slot out of place and the total at 411.
    """
    assert RECORD_STARTS[3] + DEVICES * RECORD_WIDTHS[3] == EP_START
    assert EP_START + EP_COUNTERS + EP_POINTER_TABLES == TAIL_START
    assert TAIL_START + 10 == SIGNAL_START
    assert SIGNAL_START + DEVICES == TOTAL_SLOTS
    types = _signature(checkpoint)
    draft = types[RECORD_STARTS[3]]
    assert str(draft.dtype) == "DType.bfloat16"
    tree = _tree(checkpoint)
    leaves = tree.children["draft"]
    assert isinstance(leaves, KVCacheParams)
    assert leaves.dtype == DType.bfloat16
    assert leaves.speculative_method == "dspark"
    # Its per-device symbolic inputs are exactly the record's eleven slots.
    assert _slots(leaves) == RECORD_WIDTHS[3]


def test_the_folded_scales_declare_the_scales_leaf_axes(
    checkpoint: Path,
) -> None:
    """The indexer's ``kv_scales`` slot carries the scales group's own axes.

    Mach checks that slot against the *scales* group's pool --
    ``[arity, layers, page_size, kv_heads, head_dim]`` -- not against its
    owner's, and the scales' head dim is one, since a scale covers a whole key
    row. Getting it wrong is a boot refusal from ``check_kv_scales_axes``.
    """
    types = _signature(checkpoint)
    base = RECORD_STARTS[2]
    # Device 0's record: the scales triple sits between page_stride and the
    # dispatch pair, at +6, +7 and +8.
    scales = types[base + 6]
    assert str(scales.dtype) == "DType.float32", scales
    # Drop the leading page-count dim, which is dynamic.
    axes = [getattr(dim, "value", dim) for dim in scales.shape[1:]]
    tree = _tree(checkpoint)
    target = tree.children["target"]
    assert isinstance(target, MultiKVCacheParams)
    indexer = target.children[INDEXER_GROUP]
    assert isinstance(indexer, KVCacheParams)
    assert axes == [
        1,  # arity: one key plane
        int(indexer.num_layers),
        PAGE_SIZE,
        1,  # one KV head: an MLA latent
        1,  # head_dim / granularity: one scale per key row
    ]
    # The page stride and the page table follow it.
    assert str(types[base + 7].dtype) == "DType.int64"
    assert str(types[base + 8].dtype) == "DType.uint32"


def test_the_tail_is_the_sampling_run_then_the_signals(
    checkpoint: Path,
) -> None:
    """Slots 401-410 are the sampling and bitmask run; 411-418 the signals.

    No penalty input exists anywhere in this tail: Mistral Large 4 refuses a
    penalized request at admission, so a slot for one would shift the signals.
    """
    types = _signature(checkpoint)
    tail = [str(t.dtype) for t in types[TAIL_START:SIGNAL_START]]
    assert tail == [
        "DType.int64",  # draft_tokens
        "DType.uint64",  # seed
        "DType.float32",  # temperature
        "DType.int64",  # top_k
        "DType.int64",  # max_k
        "DType.float32",  # top_p
        "DType.float32",  # min_top_p
        "DType.int32",  # pinned_bitmask
        "DType.int64",  # bitmask_wait_payload
        "DType.int32",  # device_bitmask_scratch
    ]
    assert [str(t.dtype) for t in types[SIGNAL_START:]] == [
        "DType.uint8"
    ] * DEVICES


def test_the_ep_tail_sits_between_the_kv_tree_and_the_sampling_run(
    checkpoint: Path,
) -> None:
    """Slots 379-400 are the collective buffers, at Mach's own positions.

    EP8 at one attention replica is not a data-parallel deployment, so the
    graph declares no host offsets, no replica splits and no batch context
    lengths -- only the target's collectives, in the order the engine binds
    them: two groups of atomic counters (one int32 buffer per rank each), then
    the six symmetric-heap pointer tables (one uint64 row of eight pointers
    each).
    """
    types = _signature(checkpoint)
    counters = types[EP_START : EP_START + EP_COUNTERS]
    pointers = types[EP_START + EP_COUNTERS : TAIL_START]
    assert len(counters) == EP_COUNTERS
    assert len(pointers) == EP_POINTER_TABLES
    for index, counter in enumerate(counters):
        assert str(counter.dtype) == "DType.int32", (index, counter)
        # Group-major over the ranks: two counters per device.
        assert counter.device == DeviceRef.GPU(index % DEVICES), (
            index,
            counter,
        )
    for index, table in enumerate(pointers):
        assert str(table.dtype) == "DType.uint64", (index, table)
        assert table.device == DeviceRef.CPU(), (index, table)
        assert table.rank == 1 and int(table.shape[0]) == DEVICES, table
    # The same 22 slots the manager announces: the number is MAX's, not this
    # test's.
    assert (
        len(_ep_manager(checkpoint).input_types())
        == EP_COUNTERS + EP_POINTER_TABLES
    )
    # Each counter is sized from the formula the manager applies to this
    # config's expert count, so a config change that moved the width would
    # have to move this test with it (Mach sizes the boot's handshake off the
    # sidecar's static shape).
    config = _ep_manager(checkpoint).config
    assert config.moe_dim == 0, (
        "the fused-MoE reserve is not part of this arm's counters"
    )
    assert EP_COUNTER_STATIC_SHAPE == get_ep_local_sync_counters_size(
        config.n_experts // config.n_gpus_per_node
    )
    for counter in counters:
        assert int(counter.shape[0]) == EP_COUNTER_STATIC_SHAPE, counter
    # And none of the data-parallel scaffolding came with them: the head is
    # three slots and the KV tree starts at 3, where a distributed signature
    # would have put the replica splits. A record's own host metadata is the
    # only thing wider than one slot ahead of the tail -- the dispatcher's
    # partition tables and the mixed-step split record it declares last -- and
    # every record carries the same set of it. A replica-split vector would be
    # another wide host vector, on the replica axis.
    assert len(types[:EP_START]) == 3 + DEVICES * sum(RECORD_WIDTHS), (
        "the head and the KV tree, with nothing between them"
    )
    assert types[3].device == DeviceRef.GPU(0), "the first KV record's blocks"
    # ``Shape`` holds ``Dim`` objects and a static dim is not an ``int``, so the
    # static widths come from ``static_dims``.
    wide = [
        (index, entry)
        for index, entry in enumerate(types[:EP_START])
        if entry.device == DeviceRef.CPU()
        and any(size > 1 for size in entry.shape.static_dims)
    ]
    per_record = DEVICES * len(RECORD_WIDTHS)
    assert wide, "a paged record declares its host metadata"
    assert len(wide) % per_record == 0, wide
    stride = len(wide) // per_record
    shapes = [tuple(entry.shape.static_dims) for _, entry in wide]
    assert shapes == shapes[:stride] * per_record, (
        f"every record carries the same host metadata, got {shapes}"
    )
    # Each record ends with the mixed-step split record, which is the batch's
    # and not the device's: the last wide host vector of a record is its
    # ``uint32[MIXED_STEP_SPLIT_SLOTS]``.
    assert shapes[stride - 1] == (MIXED_STEP_SPLIT_SLOTS,), shapes
    assert wide[stride - 1][1].dtype == DType.uint32, wide[stride - 1]


def test_the_ep_tail_is_declared_only_by_an_ep_build(checkpoint: Path) -> None:
    """The adapter names the tail from the target's manager, not the flag.

    A build without expert parallelism declares none of the buffers and keeps
    the 397-slot shape Mach pins for that arm; the flag alone does not add
    them.
    """
    arch = _pairing().adapter_arch
    assert spec_adapters.ep_tail(arch, _ep_target(checkpoint))
    assert not spec_adapters.ep_tail(
        arch, cast(Any, SimpleNamespace(ep_batch_manager=None))
    )
    assert len(_signature(checkpoint, ep=False)) == TOTAL_SLOTS - (
        EP_COUNTERS + EP_POINTER_TABLES
    )


def test_the_adapter_declares_the_block_abi() -> None:
    """The declarations the layout above depends on.

    The rope layers' latent is the driver's primary leaf, the acceptance never
    relaxes into a reasoning phase (which would add a slot), the trace appends
    the mesh's signals last (the layout's trailing placement), and a sampled
    draft proposal is refused rather than silently served greedily.
    """
    arch = _pairing().adapter_arch
    assert arch == "MistralLarge4ForConditionalGeneration_ModuleV3"
    assert spec_adapters.primary_kv(arch, "dspark") == ROPE_MLA_GROUP
    assert not spec_adapters.relaxed_thinking_acceptance(arch)
    assert spec_adapters.signal_buffer_index(arch) is None
    assert not spec_adapters.prepares_every_batch(arch)
    spec_adapters.check(
        arch,
        SpeculativeConfig(
            speculative_method="dspark", num_speculative_tokens=3
        ),
        True,
    )
    with pytest.raises(ValueError, match="argmax"):
        spec_adapters.check(
            arch,
            SpeculativeConfig(
                speculative_method="dspark",
                num_speculative_tokens=3,
                draft_proposal="sampled",
            ),
            True,
        )


def test_the_draft_architecture_resolves_from_the_targets_checkpoint() -> None:
    """``checkpoint_drafts`` names a draft arch that builds its own module.

    Spec decoding finds the in-checkpoint drafter only through the target's
    mapping, and pairs it only if that arch builds a module itself (no pipeline
    model, a ``build_module`` of its own).
    """
    assert dict(mistral_large4_modulev3_arch.checkpoint_drafts) == {
        "dspark": "DSparkMistralLarge4DraftModel"
    }
    arch = spec_decode_architecture(
        "MistralLarge4ForConditionalGeneration",
        "dspark",
        None,
        prefer_module_v3=True,
    )
    assert arch is not None
    assert arch.name == "MistralLarge4ForConditionalGeneration_ModuleV3+dspark"
    assert arch.spec_decode_graph_names["dspark"] == (
        "UnifiedDSparkMistralLarge4Module"
    )
    # Only the ``_ModuleV3`` spelling is registered: a draft lookup that does
    # not prefer it resolves through the registry's single-variant fallback, so
    # ``checkpoint_drafts``' bare name still finds the arch.
    draft = ARCH_LOOKUP.find("DSparkMistralLarge4DraftModel")
    assert draft is not None
    assert draft.name == "DSparkMistralLarge4DraftModel_ModuleV3"
    resolved = ARCH_LOOKUP.find(draft.name)
    assert resolved is not None and resolved.name == draft.name


def test_the_fused_architecture_refuses_images_per_request() -> None:
    """A text-only pairing of an image-taking target fails the one request.

    The pairing is text-only -- the fused DSpark graph encodes no images,
    and the target builds no ``build_vision_graph`` for it -- but it keeps
    the base vision tokenizer, so an image request would tokenize into a
    context the fused graph cannot encode. Without the validator the
    request reaches batch preparation, whose refusal raises the worker
    rather than failing the request; with it, the tokenizer's
    admission-time ``InputError`` fails that request alone, the same
    per-request error the HTTP router maps to a 4xx.
    """
    fused = spec_decode_architecture(
        "MistralLarge4ForConditionalGeneration",
        "dspark",
        None,
        prefer_module_v3=True,
    )
    assert fused is not None
    assert fused.input_modalities == {InputModality.TEXT}
    assert validate_no_images in fused.context_validators
    # The base keeps serving images; only the pairing refuses them.
    assert (
        validate_no_images
        not in mistral_large4_modulev3_arch.context_validators
    )
    assert InputModality.IMAGE in mistral_large4_modulev3_arch.input_modalities

    context = MistralLarge4VisionContext.new_padding_context(
        max_length=32, model_name="m"
    )
    # No image: the pairing serves the request.
    validate_no_images(context)
    context.images = [cast(Any, SimpleNamespace())]
    with pytest.raises(InputError, match="serves text only"):
        validate_no_images(context)


def test_the_target_pins_the_dspark_block_width() -> None:
    """The checkpoint's block width is the draft width, and nothing else is.

    The target's latent leaf and the drafter's reserve the same number of draft
    slots per step, and a cache tree refuses two leaves that disagree
    (``_agreed_pool``: ``cache_params.py``). An unset
    ``--num-speculative-tokens`` therefore has to resolve to the block's width
    through this hook rather than stay unset -- a target leaf at zero against a
    drafter at three is the refusal -- and a narrower step has to be refused
    here, where the message can say why.
    """
    width = mistral_large4_modulev3_arch.checkpoint_draft_width
    assert width is not None
    unset = SpeculativeConfig(speculative_method="dspark")
    assert unset.num_speculative_tokens is None
    assert width(unset, None, None) == 3
    named = SpeculativeConfig(
        speculative_method="dspark", num_speculative_tokens=3
    )
    assert width(named, None, None) == 3
    with pytest.raises(ValueError, match="drafts exactly"):
        width(
            SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=1
            ),
            None,
            None,
        )
    # The fused architecture is the base's, renamed, so the pairing the pipeline
    # actually builds carries the hook too.
    fused = spec_decode_architecture(
        "MistralLarge4ForConditionalGeneration",
        "dspark",
        None,
        prefer_module_v3=True,
    )
    assert fused is not None
    assert fused.checkpoint_draft_width is width


def test_the_kv_tree_holds_the_four_leaves_in_record_order(
    checkpoint: Path,
) -> None:
    """The tree's children are the layout's records, in the same order.

    Mach maps each record to a ``kv_cache_configs`` group by position, so the
    target's three leaves must come before the drafter's and the indexer last
    of the target's (it folds the scales group).
    """
    target = _tree(checkpoint).children["target"]
    assert isinstance(target, MultiKVCacheParams)
    assert list(target.children) == [
        ROPE_MLA_GROUP,
        NOPE_MLA_GROUP,
        INDEXER_GROUP,
    ]
    assert list(_tree(checkpoint).children) == ["target", "draft"]


def test_the_spec_the_driver_builds_declares_the_ep_tail(
    checkpoint: Path,
) -> None:
    """The signature an EP target declares comes from the driver's own spec.

    ``_driver_call`` builds its spec through
    :func:`~max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model.
    _input_type_spec`, which asks the target's adapter; this file's signature
    follows. Dropping that term would leave a spec that is false here and a
    graph 22 slots short of Mach's layout, with nothing else changed -- so the
    flag is asserted where the driver sets it, not only on the adapter.
    """
    spec = _input_spec(checkpoint)
    assert spec.ep_tail, "an EP target declares its collectives"
    # The width the batch processor checks a staging batch against is the one
    # the graph recorded, and it is the tail's own slot count.
    assert spec.ep_tail_arity == EP_COUNTERS + EP_POINTER_TABLES
    # ... and no data-parallel scaffolding with them.
    assert not spec.distributed
    assert spec.data_parallel_degree == 1
    assert not spec.include_in_thinking_phase
    assert spec.signal_buffer_index is None
    assert spec.trace_owns_signal_buffers
    # A build without expert parallelism declares none of them.
    without = _input_spec(checkpoint, ep=False)
    assert not without.ep_tail
    assert without.ep_tail_arity == 0
