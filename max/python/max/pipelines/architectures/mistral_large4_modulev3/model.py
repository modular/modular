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
"""The Mistral Large 4 ModuleV3 pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar

from max import functional as F
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import (
    KVCacheParamInterface,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.legacy.comm.ep import (
    EPBatchManager,
)
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor import default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3_modulev3.model import DeepseekV3Model
from .config import MistralLarge4Config, latent_cache_dtype
from .mistral_large4 import MistralLarge4
from .vision import (
    MistralLarge4Vision,
    MistralLarge4VisionConfig,
    vision_input_types,
)
from .vision_inputs import pack_images
from .weight_contract import is_vision_name


class MistralLarge4Model(DeepseekV3Model):
    """A Mistral Large 4 text-and-vision model (ModuleV3).

    Inherits the DeepSeek-V3 weight loading, mesh setup and expert-parallel
    runtime. What it overrides is the config, which selects this checkpoint's
    mixed quantization from the architecture rather than from a single
    checkpoint-wide ``quantization_config``: the routed experts are NVFP4 while
    ``wq``, ``wo``, ``indexer.wq_b``, the shared experts and layer 0 are FP8
    block-scaled and the MLA projections are BF16.

    The vision tower and projector (``vision.py``) compile as a second graph
    beside the decoder, on device 0 at any device count. The pipeline's
    vision-encoder cache drives it through the
    :class:`~max.pipelines.lib.vision_encoder_cache.SupportsVisionEncoding`
    entry points and hands the language graph the embeddings to scatter into
    the prompt's placeholder rows; a text-only or decode step passes the
    model's empty embeddings, and never runs the tower. The load parses the
    whole checkpoint through the adapter -- the vision table's BF16 check and
    the tower's Q/K rope-layout permutation run there -- and the two graphs
    take disjoint slices: the decoder binds no vision name, the vision graph
    binds only vision names.
    """

    model_config_cls: ClassVar[type[Any]] = MistralLarge4Config

    _language_export_includes_vision: ClassVar[bool] = False
    """Whether the MEF language export compiles the vision graph.

    False until the engine's vision contract lands: the engine's
    ML4 slot map declares no vision inputs, so a language MEF whose
    decoder carries the vision-merge tail would abort graph-capture
    warmup with an arity error. MAX serve is unaffected -- it includes
    the tower through ``_include_vision_graph`` whatever this says.
    """

    @cached_property
    def _has_vision_tower(self) -> bool:
        """Whether this checkpoint carries a vision tower at all.

        A checkpoint without a ``vision_config`` is refused at config
        creation (see :meth:`_create_model_config`): this architecture
        tokenizes image placeholders and scatters image embeddings into
        them, so a tower-less checkpoint has no servable text-only
        fallback -- it is a checkpoint error, not a degraded mode.
        """
        return (
            getattr(self.huggingface_config, "vision_config", None) is not None
        )

    @cached_property
    def _vision_config(self) -> MistralLarge4VisionConfig:
        """The vision graph's geometry, off the pipeline's HF config."""
        return MistralLarge4VisionConfig.from_hf_config(self.huggingface_config)

    def _include_vision_graph(self, model_config: Any) -> bool:
        """Whether this build compiles the tower and the vision ABI tail.

        MAX serve includes both for every checkpoint with a tower. The MEF
        exporter replaces this hook for its language-only build
        (``gen_mef._language_only_compile``), which is how the export
        keeps the text-only language ABI Mach's ML4 slot map declares
        until the engine-side vision contract lands.

        Args:
            model_config: The config this build was finalized with, unread.
        """
        del model_config
        return self._has_vision_tower

    @property
    def vision_model(self) -> CompiledCallable[..., Any]:
        """The compiled vision tower."""
        vision = self.compiled_modules[GraphRole.VISION]
        assert isinstance(vision, CompiledCallable)
        return vision

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: Any,
        devices: list[DeviceRef],
        kv_cache_config: Any,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """The KV cache tree this model was traced with.

        The framework derives ``cache_dtype`` from the quantization encoding,
        which for this family would be bf16 while the decoder's ops are told
        the config tree's FP8; the tree the graph's inputs are flattened from
        has to be the tree the model builds, so both resolve it through
        :func:`~.config.latent_cache_dtype`.
        """
        del cache_dtype
        return MistralLarge4Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=latent_cache_dtype(kv_cache_config.kv_cache_format),
        )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        if not self._has_vision_tower:
            # Refused while the pipeline loads, not on the first request:
            # the vision tokenizer would expand image placeholders the
            # language graph could never fill, and the empty-image drive
            # every batch runs would read the missing vision geometry.
            raise ValueError(
                "this Mistral Large 4 checkpoint declares no vision_config,"
                " but the ModuleV3 architecture serves text and images"
                " together: its tokenizer expands image placeholders and"
                " its decoder scatters image embeddings into them, so a"
                " checkpoint without a vision tower cannot be served"
            )
        model_config = MistralLarge4Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or model_config.max_batch_context_length
        )
        model_config.quant_config = model_config.nvfp4_config
        n_devices = len(self.devices)
        dp_degree = self.pipeline_config.model.data_parallel_degree
        if dp_degree > 1 and dp_degree != n_devices:
            raise NotImplementedError(
                f"data_parallel_degree={dp_degree} must equal the device"
                f" count ({n_devices}) on Mistral Large 4 ModuleV3."
            )
        axis_name = "dp" if dp_degree > 1 else "tp"
        model_config.mesh = DeviceMesh(
            tuple(self.devices), (n_devices,), (axis_name,)
        )
        # The pipeline's decoder declares the vision-merge inputs whenever
        # this build includes the vision graph: its batches carry image
        # embeddings through the vision-encoder cache, empties on a
        # text-only or decode step. Callers that build the module outside
        # this model (the fused DSpark graph) leave the flag off and keep
        # the text-only ABI, and the exporter's language-only build turns
        # the same hook off, so its decoder keeps the text-only ABI too.
        model_config.encodes_images = self._include_vision_graph(model_config)
        return model_config

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        del state_dict, model_config
        # The checkpoint stores its non-quantized tensors -- norms, embeddings,
        # the MLA projections, the router, the head -- in bfloat16.
        return DType.bfloat16

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        """Loads the whole checkpoint, and keeps it for ``graph_specs``.

        The vision subtree rides the same parse the text weights do: the
        adapter's vision table runs its "unquantized BF16" check here and
        permutes the tower's 60 interleaved-RoPE Q/K rows, and the vision
        graph's spec takes its slice of the result. The two graphs bind
        disjoint name sets, which :meth:`_prepare_state_dict` enforces for
        the decoder at bind time.

        This override replaces three one-line base behaviours, all spelled
        out here so a base-class change is not silently skipped: the
        ``DeepseekV2Model`` and ``DeepseekV3Model`` safetensors-only
        guards, and ``ModuleV3PipelineModelWithKVCache._load_state_dict``'s
        ``self._checkpoint`` assignment.

        Raises:
            ValueError: If the checkpoint is not safetensors, the only
                format the DeepSeek-V3 loader reads.
        """
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are supported in Mistral Large 4."
            )
        self._checkpoint = parse_state_dict_from_weights(
            self.pipeline_config,
            self.weights,
            self.adapter,
            hf_config=self._hf_config_for_weights(),
        )
        return self._checkpoint

    @override
    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        """The language graph binds only the text weights.

        The load parses the vision subtree too -- the vision graph's spec
        takes it -- so this bind-time guard is what keeps the decoder from
        binding a weight it does not declare.
        """
        del model_config
        return {
            name: weight
            for name, weight in state_dict.items()
            if not is_vision_name(name)
        }

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        assert model_config.mesh is not None
        with default_device(model_config.mesh):
            module = MistralLarge4(
                model_config, self.kv_params, self._ep_batch_manager
            )
        # The output modes are the caller's, not the module's: a verify asks
        # for every row's logits and, with taps declared, for the drafter's
        # concatenated residual streams.
        module.language_model.return_logits = self.return_logits
        module.language_model.return_hidden_states = self.return_hidden_states
        return module

    @override
    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """The decoder, plus the vision tower on device 0.

        The base builds and stages the decoder (and initializes the
        expert-parallel runtime its input tail depends on); the vision
        graph rides beside it as its own role, compiled on its own.
        """
        specs = dict(super().graph_specs(model_config))
        if self._include_vision_graph(model_config):
            specs[GraphRole.VISION] = self._vision_tower_spec()
        return specs

    def _vision_tower_spec(self) -> GraphSpec:
        """The vision tower and projector, on device 0.

        The tower is never tensor-parallel: it runs once per batch on the
        first device, whatever the decoder's mesh is, and the language
        graph broadcasts its rows. Its weights are the vision slice of the
        parsed checkpoint, under the vision table's module names.
        """
        config = self._vision_config
        with F.lazy(), default_dtype(config.dtype):
            vision_nn = MistralLarge4Vision(config)
            vision_nn.to(self.devices[0])
        return GraphSpec(
            vision_nn,
            type(vision_nn).__qualname__,
            vision_input_types(config, self.device_refs[0]),
            weights={
                name: weight
                for name, weight in self._checkpoint.items()
                if is_vision_name(name)
            },
        )

    @override
    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        """The decoder's ABI: the base's inputs, then the vision tail.

        The merge inputs follow the KV cache's leaves and the EP buffers,
        which is where :meth:`.mistral_large4.MistralLarge4.forward` peels
        them and where ``DeepseekV3Inputs.buffers`` packs them. Only the
        pipeline's decoder declares them; a caller that built its module
        without ``encodes_images`` keeps the text-only ABI.
        """
        input_types = tuple(super()._get_compile_input_types(model_config))
        if not model_config.encodes_images:
            return input_types
        device = self.device_refs[0]
        return (
            *input_types,
            TensorType(
                self._vision_config.dtype,
                ["num_image_tokens", model_config.hidden_size],
                device=device,
            ),
            TensorType(DType.int32, ["total_image_tokens"], device=device),
        )

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer]:
        """Packs the images the vision encoder cache selected for this step.

        The tower runs on the first device, so that is where the patch
        rows land.
        """
        return pack_images(
            [image for _, images in selection for image in images],
            self._vision_config,
            devices[0],
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Encodes the packed images into one row per placeholder.

        The tower emits on device 0; every other device takes a copy, which
        is what the encoder cache's per-device bookkeeping and the
        language graph's device-0 input expect.
        """
        if packed is None:
            packed = self.pack_vision_inputs(selection, devices)
        output = self.vision_model(*packed).driver_tensor
        assert isinstance(output, Buffer)
        return VisionEncodeResult(
            embeddings=[output.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row embeddings, for batches with no new image.

        Built once for the pipeline's own devices, which every text-only
        and decode step asks for; any other device list gets fresh
        empties sized the same, for those devices.
        """
        if devices == self.devices:
            return self._empty_vision_embeddings
        return create_empty_image_embeddings(
            devices,
            self._vision_config.hidden_size,
            self._vision_config.dtype,
        )

    @cached_property
    def _empty_vision_embeddings(self) -> list[Buffer]:
        return create_empty_image_embeddings(
            self.devices,
            self._vision_config.hidden_size,
            self._vision_config.dtype,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[MistralLarge4, MistralLarge4Config]:
        """Builds the target for a DSpark pairing, without its weights.

        The module compiles with the weights :meth:`checkpoint_weights`
        returns. The verify gathers the logit rows the acceptance scores and
        taps the layers the drafter's projection was trained on, so the target
        keeps the variable logits and returns its selected layers; the
        last-token head is skipped, which costs the verify nothing and would
        otherwise change the output tuple the adapter indexes.

        Args:
            context: The load context, which names the KV tree and the taps.

        Raises:
            ValueError: If the checkpoint resolves no taps, or the pairing's
                taps are not the ones the checkpoint declares.
            NotImplementedError: Under data-parallel attention, whose VARIABLE
                logits this trunk does not implement.
        """
        pipeline_config = context.pipeline_config
        dp_degree = pipeline_config.model.data_parallel_degree
        ep_size = pipeline_config.runtime.ep_size
        if dp_degree > 1:
            raise NotImplementedError(
                "Mistral Large 4's speculative graph serves data-parallel"
                " attention through VARIABLE logits, which this trunk does not"
                " implement yet. Export at data_parallel_degree=1."
            )
        max_seq_len = context.max_seq_len
        kv_params = context.kv_params
        if kv_params is not None:
            assert isinstance(kv_params, MultiKVCacheParams)
            max_seq_len += spec_decode_cache_slack(kv_params)
        config = MistralLarge4Config.initialize(
            pipeline_config, max_seq_len=max_seq_len
        )
        if kv_params is not None:
            config.kv_params = kv_params
        layers = list(context.target_layer_ids)
        if not layers:
            raise ValueError(
                "Mistral Large 4 verifies for its DSpark drafter, so the"
                " target's tap layers must be resolved; the checkpoint"
                " declares none"
            )
        if layers != list(config.target_layer_ids):
            raise ValueError(
                "Mistral Large 4's DSpark drafter reads the taps the"
                f" checkpoint declares ({list(config.target_layer_ids)}); the"
                f" pairing captured {layers}."
            )
        config.quant_config = config.nvfp4_config
        # The fused DSpark graph traces ``verify`` and declares no vision
        # slots of its own, so its target keeps the text-only ABI; a
        # pairing whose draft also takes images passes True here.
        config.encodes_images = context.encodes_images
        config.data_parallel_degree = dp_degree
        config.max_batch_context_length = (
            context.max_batch_total_tokens or config.max_batch_context_length
        )
        devices = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]
        config.mesh = DeviceMesh(
            tuple(ref.to_device() for ref in devices), (len(devices),), ("tp",)
        )
        # Expert parallelism is the deployment's, as in the base build: every
        # rank holds the same rows and dispatches its share of the experts
        # through one collective group, so the fused graph takes the same
        # buffers the base graph does. The config's own decision table sizes
        # it, so the two graphs cannot disagree about the group.
        ep_batch_manager: EPBatchManager | None = None
        if ep_size > 1:
            if ep_size % len(devices) != 0:
                raise ValueError(
                    f"ep_size={ep_size} must be divisible by the number of"
                    f" GPUs on this node ({len(devices)}); for single-node set"
                    f" ep_size={len(devices)}."
                )
            config.ep_config = config.expert_parallel_config(
                ModuleRuntime(
                    mesh=config.mesh,
                    data_parallel_degree=dp_degree,
                    max_batch_input_tokens=(
                        pipeline_config.runtime.max_batch_input_tokens
                    ),
                    max_batch_total_tokens=context.max_batch_total_tokens,
                    ep_size=ep_size,
                    ep_use_allreduce=pipeline_config.runtime.ep_use_allreduce,
                )
            )
            context.start_ep(config.ep_config)
            ep_batch_manager = EPBatchManager(config.ep_config)
        with (
            F.lazy(),
            default_dtype(DType.bfloat16),
            default_device(config.mesh),
        ):
            module = MistralLarge4(config, config.kv_params, ep_batch_manager)
        module.language_model.return_logits = context.return_logits
        module.language_model.return_hidden_states = (
            context.return_hidden_states
        )
        # A verify never needs the last-token head: the acceptance reads the
        # variable rows, and leaving the head on would add a `[batch, vocab]`
        # projection per step and shift the output tuple.
        module.language_model.emit_last_token_logits = (
            context.return_logits is not ReturnLogits.VARIABLE
        )
        return module, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: MistralLarge4Config
    ) -> dict[str, Any]:
        """Returns the target's weights, by module name.

        The vision subtree belongs to the vision graph, not this one, so it
        is filtered out before the adapter materializes it; the checkpoint's
        own adapter then renames every text tensor and drops the drafter's
        ``mtp_dspark.*`` subtree, which the fused module's draft half binds
        instead.

        Raises:
            ValueError: If the pairing resolved no safetensors adapter.
        """
        if context.adapter is None:
            raise ValueError(
                "Mistral Large 4 ModuleV3 needs its safetensors weight"
                " adapter to compile"
            )
        adapted = dict(
            context.adapter(
                {
                    name: weight
                    for name, weight in context.weights.items()
                    if not is_vision_name(name)
                },
                huggingface_config=(
                    context.pipeline_config.model.huggingface_config
                ),
                pipeline_config=context.pipeline_config,
            )
        )
        return adapted
