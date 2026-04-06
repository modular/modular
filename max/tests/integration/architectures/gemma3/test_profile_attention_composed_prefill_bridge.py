from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    KVCacheParams,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    q_rms_norm_fused_qk_ragged_rope,
    rms_norm_key_cache,
)
from max.nn.kv_cache import (
    AttentionDispatchResolver,
    KVCacheInputsPerDevice,
    build_max_lengths_tensor,
    unflatten_ragged_attention_inputs,
)
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.attention import (
    Gemma3Attention as MaxGemma3Attention,
)
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack


PAGE_SIZE = 128
EPS = 1e-6
DEFAULT_LAYER_IDX = 0
DEFAULT_LAYER_TYPE = "local"
Q_NORM_STD = 0.68
K_NORM_STD = 0.793
Q_PROJ_STD = 0.0284
K_PROJ_STD = 0.0309
V_PROJ_STD = 0.0309
O_PROJ_STD = 0.0237
WARMUP_ITERS = 20
TIMED_ITERS = 50
PREFILL_SHAPES = (
    (1, 11),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (2, 2048),
)
PLACEHOLDER_CACHE_LEN = 1


class CorrectnessMismatch(AssertionError):
    def __init__(self, stage: str, details: str):
        super().__init__(f"{stage} mismatch: {details}")
        self.stage = stage
        self.details = details


def _resolve_layer_metadata(config: dict[str, Any]) -> tuple[int, str]:
    layer_idx = int(
        os.environ.get("PROFILE_ATTENTION_LAYER_IDX", str(DEFAULT_LAYER_IDX))
    )
    num_hidden_layers = int(config["num_hidden_layers"])
    assert 0 <= layer_idx < num_hidden_layers, (
        f"layer_idx={layer_idx} must be in [0, {num_hidden_layers})"
    )
    layer_type = (
        "local"
        if bool((layer_idx + 1) % config["sliding_window_pattern"])
        else "global"
    )
    expected_layer_type = os.environ.get(
        "PROFILE_ATTENTION_LAYER_TYPE", DEFAULT_LAYER_TYPE
    )
    assert (
        layer_type == expected_layer_type
    ), f"expected {expected_layer_type} layer, got {layer_type} for layer_idx={layer_idx}"
    return layer_idx, layer_type


def _resolve_kv_num_layers(config: dict[str, Any], layer_idx: int) -> int:
    min_num_layers = layer_idx + 1
    kv_num_layers = int(
        os.environ.get("PROFILE_ATTENTION_KV_NUM_LAYERS", str(min_num_layers))
    )
    assert kv_num_layers >= min_num_layers, (
        f"kv_num_layers={kv_num_layers} must cover layer_idx={layer_idx}"
    )
    assert kv_num_layers <= int(config["num_hidden_layers"]), (
        "kv_num_layers cannot exceed the model layer count "
        f"({config['num_hidden_layers']})"
    )
    return kv_num_layers


def _resolve_flash_mask_config(
    *,
    use_local: bool,
    default_local_window_size: int,
) -> tuple[MHAMaskVariant, int, str]:
    mask_override = os.environ.get(
        "PROFILE_FLASH_ATTENTION_MASK_VARIANT", "layer_default"
    ).strip()
    local_window_override = os.environ.get(
        "PROFILE_FLASH_ATTENTION_LOCAL_WINDOW_SIZE"
    )

    if local_window_override is None:
        local_window_size = default_local_window_size if use_local else -1
    else:
        local_window_size = int(local_window_override)

    if mask_override in ("", "layer_default"):
        return (
            (
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if use_local
                else MHAMaskVariant.CAUSAL_MASK
            ),
            local_window_size,
            "layer_default",
        )

    if mask_override == "causal":
        return MHAMaskVariant.CAUSAL_MASK, -1, mask_override

    if mask_override == "sliding_window_causal":
        return (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
            local_window_size,
            mask_override,
        )

    raise ValueError(
        "PROFILE_FLASH_ATTENTION_MASK_VARIANT must be one of "
        "{'layer_default', 'causal', 'sliding_window_causal'}, got "
        f"{mask_override!r}"
    )


def _resolve_output_mode() -> tuple[str, bool]:
    output_mode = os.environ.get(
        "PROFILE_ATTENTION_COMPOSED_OUTPUT_MODE", "attn_and_output"
    ).strip()
    if output_mode == "attn_and_output":
        return output_mode, True
    if output_mode == "attention_output_only":
        return output_mode, False
    raise ValueError(
        "PROFILE_ATTENTION_COMPOSED_OUTPUT_MODE must be one of "
        "{'attn_and_output', 'attention_output_only'}, got "
        f"{output_mode!r}"
    )


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _make_weight_registry(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    q_dim = config["head_dim"] * config["num_attention_heads"]
    kv_dim = config["head_dim"] * config["num_key_value_heads"]
    hidden_size = config["hidden_size"]
    return {
        "k_norm.weight": torch.randn(
            config["head_dim"], dtype=torch.bfloat16
        )
        * K_NORM_STD,
        "k_proj.weight": torch.randn(
            kv_dim, hidden_size, dtype=torch.bfloat16
        )
        * K_PROJ_STD,
        "o_proj.weight": torch.randn(
            hidden_size, q_dim, dtype=torch.bfloat16
        )
        * O_PROJ_STD,
        "q_norm.weight": torch.randn(
            config["head_dim"], dtype=torch.bfloat16
        )
        * Q_NORM_STD,
        "q_proj.weight": torch.randn(
            q_dim, hidden_size, dtype=torch.bfloat16
        )
        * Q_PROJ_STD,
        "v_proj.weight": torch.randn(
            kv_dim, hidden_size, dtype=torch.bfloat16
        )
        * V_PROJ_STD,
    }


def _make_placeholder_text_context(max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.zeros(PLACEHOLDER_CACHE_LEN, dtype=np.int64)),
    )


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _row_offsets_array(batch_size: int, seq_len: int) -> np.ndarray:
    return np.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        dtype=np.uint32,
    )


def _prefill_run_name(
    layer_type: str,
    layer_idx: int,
    batch_size: int,
    seq_len: int,
    output_mode: str,
) -> str:
    suffix = (
        "_cache0_step0_attention_composed_bridge"
        if output_mode == "attn_and_output"
        else "_cache0_step0_attention_composed_bridge_output_only"
    )
    return (
        f"prefill_{layer_type}_layer{layer_idx}_bs{batch_size}_seq{seq_len}"
        f"{suffix}"
    )


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    batch_size: int,
    seq_len: int,
    use_fused: bool,
    layer_idx: int,
    layer_type: str,
    observe_attn_out: bool,
) -> Any:
    device_ref = DeviceRef.GPU()
    max_seq_len = seq_len + PLACEHOLDER_CACHE_LEN + 256
    attention = MaxGemma3Attention(
        rope_global=Llama3RotaryEmbedding(
            config["hidden_size"],
            config["num_attention_heads"],
            config["rope_theta"],
            max_seq_len,
            interleaved=False,
            head_dim=config["head_dim"],
        ),
        rope_local=Llama3RotaryEmbedding(
            config["hidden_size"],
            config["num_attention_heads"],
            config["rope_local_base_freq"],
            max_seq_len,
            interleaved=False,
            head_dim=config["head_dim"],
        ),
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        hidden_size=config["hidden_size"],
        kv_params=kv_params,
        layer_idx=layer_idx,
        dtype=DType.bfloat16,
        devices=[device_ref],
        qk_norm_eps=EPS,
        sliding_window_pattern=config["sliding_window_pattern"],
        local_window_size=config["sliding_window"],
        has_bias=bool(config["attention_bias"]),
    )
    attention.load_state_dict(weight_registry)

    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", config["hidden_size"]],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_name = (
        "Gemma3AttentionComposedPrefillBridge"
        f"{layer_type.title()}FusedBS{batch_size}Seq{seq_len}"
        if use_fused
        else "Gemma3AttentionComposedPrefillBridge"
        f"{layer_type.title()}BaselineBS{batch_size}Seq{seq_len}"
    )
    graph_name += "Observed" if observe_attn_out else "OutputOnly"

    with Graph(
        graph_name,
        input_types=(input_type, input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        x, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        graph_layer_idx = ops.constant(
            layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        total_seq_len = x.tensor.shape[0]

        xq = fused_qkv_ragged_matmul(
            kv_params,
            input=x.tensor,
            wqkv=attention.wqkv,
            bias=attention.wqkv_bias,
            input_row_offsets=input_row_offsets.tensor,
            kv_collection=kv_collection,
            layer_idx=graph_layer_idx,
            n_heads=attention.n_heads,
        )
        xq = xq.reshape((-1, attention.n_heads, kv_params.head_dim))

        use_local = bool((layer_idx + 1) % attention.sliding_window_pattern)
        assert use_local == (layer_type == "local")
        rope = attention.rope_local if use_local else attention.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

        if use_fused:
            xq = q_rms_norm_fused_qk_ragged_rope(
                kv_params,
                xq,
                input_row_offsets.tensor,
                kv_collection,
                freqs_cis,
                attention.q_norm.weight.cast(kv_params.dtype).to(device_ref),
                attention.k_norm.weight.cast(kv_params.dtype).to(device_ref),
                EPS,
                graph_layer_idx,
                weight_offset=1.0,
                interleaved=False,
            )
        else:
            rms_norm_key_cache(
                kv_params,
                kv_collection=kv_collection,
                gamma=attention.k_norm.weight.cast(kv_params.dtype).to(
                    device_ref
                ),
                epsilon=EPS,
                layer_idx=graph_layer_idx,
                total_seq_len=total_seq_len,
                input_row_offsets=input_row_offsets.tensor,
                weight_offset=1.0,
            )
            xq = attention.q_norm(xq)
            xq = fused_qk_ragged_rope(
                kv_params,
                xq,
                input_row_offsets.tensor,
                kv_collection,
                freqs_cis,
                graph_layer_idx,
                interleaved=False,
            )

        mask_variant, local_window_size, _ = _resolve_flash_mask_config(
            use_local=use_local,
            default_local_window_size=attention.local_window_size,
        )
        attn_out = flash_attention_ragged(
            kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=graph_layer_idx,
            input_row_offsets=input_row_offsets.tensor,
            mask_variant=mask_variant,
            scale=attention.scale,
            local_window_size=local_window_size,
        )
        flattened_attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        attention_output = attention.o_proj(flattened_attn_out)
        if observe_attn_out:
            graph.output(attn_out, attention_output)
        else:
            graph.output(attention_output)

    return session.load(graph, weights_registry=attention.state_dict())


def _make_runtime_inputs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    device: Accelerator,
) -> KVCacheInputsPerDevice:
    max_cache_length = PLACEHOLDER_CACHE_LEN + seq_len
    total_num_pages = batch_size * math.ceil(max_cache_length / PAGE_SIZE)
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=batch_size,
    )

    contexts: list[TextContext] = []
    for _ in range(batch_size):
        context = _make_placeholder_text_context(max_cache_length)
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(
            context,
            replica_idx=0,
            num_steps=seq_len,
        )
        contexts.append(context)

    runtime_inputs = kv_manager.runtime_inputs(
        [contexts],
        num_steps=seq_len,
    ).inputs[0]

    dispatch_metadata = AttentionDispatchResolver(
        devices=kv_params.devices,
        is_mla=kv_params.is_mla,
        n_kv_heads_per_device=kv_params.n_kv_heads_per_device,
        num_q_heads_per_device=kv_params.num_q_heads_per_device,
        is_fp8_kv=kv_params.is_fp8_kv_dtype,
    ).resolve_for_replica(
        batch_size,
        seq_len,
        seq_len,
    )[0]

    return KVCacheInputsPerDevice(
        blocks=runtime_inputs.blocks,
        cache_lengths=_device_uint32_buffer(
            np.zeros(batch_size, dtype=np.uint32), device
        ),
        lookup_table=runtime_inputs.lookup_table,
        max_lengths=build_max_lengths_tensor(1, seq_len, seq_len),
        kv_scales=runtime_inputs.kv_scales,
        attention_dispatch_metadata=dispatch_metadata,
    )


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    return Buffer.from_dlpack(tensor)


def _make_execution_args(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: Accelerator,
    x_seed: int,
    blocks_seed: int,
) -> tuple[Any, ...]:
    runtime_inputs = _make_runtime_inputs(
        session=session,
        kv_params=kv_params,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    torch.manual_seed(x_seed)
    x = torch.randn(
        (batch_size * seq_len, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()

    return (
        Buffer.from_dlpack(x),
        _device_uint32_buffer(_row_offsets_array(batch_size, seq_len), device),
        _clone_kv_blocks(runtime_inputs.blocks, seed=blocks_seed),
        runtime_inputs.cache_lengths,
        runtime_inputs.lookup_table,
        runtime_inputs.max_lengths,
        runtime_inputs.attention_dispatch_metadata,
    )


def _to_bfloat16_tensor(value: Any) -> torch.Tensor:
    return from_dlpack(value).to(torch.bfloat16).contiguous()


def _assert_close(
    stage: str,
    baseline: torch.Tensor,
    fused: torch.Tensor,
) -> None:
    try:
        torch.testing.assert_close(
            baseline,
            fused,
            rtol=2 * torch.finfo(torch.bfloat16).eps,
            atol=8 * torch.finfo(torch.bfloat16).eps,
        )
    except AssertionError as exc:
        raise CorrectnessMismatch(stage, str(exc)) from exc


def _run_correctness_check(
    *,
    baseline: Any,
    fused: Any,
    baseline_args: tuple[Any, ...],
    fused_args: tuple[Any, ...],
    layer_idx: int,
    observe_attn_out: bool,
) -> None:
    baseline_outputs = baseline.execute(*baseline_args)
    fused_outputs = fused.execute(*fused_args)
    torch.cuda.synchronize()

    if observe_attn_out:
        baseline_attn_out, baseline_output = baseline_outputs
        fused_attn_out, fused_output = fused_outputs
        _assert_close(
            "attn_out",
            _to_bfloat16_tensor(baseline_attn_out),
            _to_bfloat16_tensor(fused_attn_out),
        )
    else:
        (baseline_output,) = baseline_outputs
        (fused_output,) = fused_outputs
    _assert_close(
        "attention_output",
        _to_bfloat16_tensor(baseline_output),
        _to_bfloat16_tensor(fused_output),
    )

    baseline_blocks = _to_bfloat16_tensor(baseline_args[2])[
        :, :, layer_idx : layer_idx + 1, :, :, :
    ]
    fused_blocks = _to_bfloat16_tensor(fused_args[2])[
        :, :, layer_idx : layer_idx + 1, :, :, :
    ]
    _assert_close("kv_cache_layer_slice", baseline_blocks, fused_blocks)


def _benchmark_us(compiled: Any, args: list[tuple[Any, ...]]) -> float:
    warmup_args = args[:WARMUP_ITERS]
    timed_args = args[WARMUP_ITERS : WARMUP_ITERS + TIMED_ITERS]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / len(timed_args)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _localization_for_stage(stage: str) -> str:
    if stage == "attn_out":
        return "composed_bridge_attn_out_fails_when_tail_stays_in_graph"
    if stage == "attention_output":
        return (
            "composed_bridge_attn_out_passes_but_attention_output_fails_"
            "inside_reshape_or_o_proj"
        )
    if stage == "kv_cache_layer_slice":
        return "composed_bridge_outputs_pass_but_kv_cache_layer_slice_diverges"
    return f"composed_bridge_failed_at_{stage}"


def _output_only_localization_for_stage(stage: str) -> str:
    if stage == "attention_output":
        return "composed_bridge_output_only_fails_without_attn_out_observation"
    if stage == "kv_cache_layer_slice":
        return "composed_bridge_output_only_matches_output_but_kv_cache_diverges"
    return f"composed_bridge_output_only_failed_at_{stage}"


def _print_results(results: dict[str, Any]) -> None:
    print("GEMMA3_ATTENTION_COMPOSED_PREFILL_BRIDGE_PROFILE_START")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("GEMMA3_ATTENTION_COMPOSED_PREFILL_BRIDGE_PROFILE_END")


def test_profile_attention_composed_prefill_bridge() -> None:
    config = _load_text_config()
    layer_idx, layer_type = _resolve_layer_metadata(config)
    kv_num_layers = _resolve_kv_num_layers(config, layer_idx)
    mask_variant, local_window_size, mask_mode = _resolve_flash_mask_config(
        use_local=(layer_type == "local"),
        default_local_window_size=int(config["sliding_window"]),
    )
    output_mode, observe_attn_out = _resolve_output_mode()
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        num_layers=kv_num_layers,
        page_size=PAGE_SIZE,
    )
    weight_registry = _make_weight_registry(config)

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "hidden_size": config["hidden_size"],
            "head_dim": config["head_dim"],
            "num_q_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "page_size": PAGE_SIZE,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "kv_num_layers": kv_num_layers,
            "mask_variant": mask_variant.name,
            "mask_mode": mask_mode,
            "local_window_size": local_window_size,
            "rope": "non-interleaved",
            "mode": (
                "prefill-ragged-attention-composed-bridge"
                if observe_attn_out
                else "prefill-ragged-attention-composed-bridge-output-only"
            ),
            "output_mode": output_mode,
            "outputs": (
                ["attn_out", "attention_output"]
                if observe_attn_out
                else ["attention_output"]
            ),
            "shapes": [
                _prefill_run_name(
                    layer_type,
                    layer_idx,
                    batch_size,
                    seq_len,
                    output_mode,
                )
                for batch_size, seq_len in PREFILL_SHAPES
            ],
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "placeholder_cache_len": PLACEHOLDER_CACHE_LEN,
        },
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        "average_speedup_ratio_vs_composed_bridge_baseline": {},
    }

    large_shape_names: list[str] = []

    for batch_size, seq_len in PREFILL_SHAPES:
        run_name = _prefill_run_name(
            layer_type, layer_idx, batch_size, seq_len, output_mode
        )
        if seq_len >= 512:
            large_shape_names.append(run_name)

        baseline = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=False,
            layer_idx=layer_idx,
            layer_type=layer_type,
            observe_attn_out=observe_attn_out,
        )
        fused = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=True,
            layer_idx=layer_idx,
            layer_type=layer_type,
            observe_attn_out=observe_attn_out,
        )

        baseline_correctness_args = _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=config["hidden_size"],
            device=device,
            x_seed=1000 + batch_size + seq_len,
            blocks_seed=2000 + batch_size + seq_len,
        )
        fused_correctness_args = _make_execution_args(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=config["hidden_size"],
            device=device,
            x_seed=1000 + batch_size + seq_len,
            blocks_seed=2000 + batch_size + seq_len,
        )
        try:
            _run_correctness_check(
                baseline=baseline,
                fused=fused,
                baseline_args=baseline_correctness_args,
                fused_args=fused_correctness_args,
                layer_idx=layer_idx,
                observe_attn_out=observe_attn_out,
            )
        except CorrectnessMismatch as exc:
            results["correctness"] = "failed"
            results["failure_stage"] = exc.stage
            results["failure_shape"] = run_name
            results["failure_message"] = exc.details
            results["localization"] = (
                _localization_for_stage(exc.stage)
                if observe_attn_out
                else _output_only_localization_for_stage(exc.stage)
            )
            _print_results(results)
            raise
        del baseline_correctness_args
        del fused_correctness_args
        torch.cuda.empty_cache()

        baseline_first_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=3000 + idx,
                blocks_seed=4000 + idx,
            )
            for idx in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        fused_first_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=3000 + idx,
                blocks_seed=4000 + idx,
            )
            for idx in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        baseline_first = _benchmark_us(baseline, baseline_first_args)
        fused_first = _benchmark_us(fused, fused_first_args)
        del baseline_first_args
        del fused_first_args
        torch.cuda.empty_cache()

        baseline_confirm_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=5000 + idx,
                blocks_seed=6000 + idx,
            )
            for idx in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        fused_confirm_args = [
            _make_execution_args(
                session=session,
                kv_params=kv_params,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=config["hidden_size"],
                device=device,
                x_seed=5000 + idx,
                blocks_seed=6000 + idx,
            )
            for idx in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        baseline_confirm = _benchmark_us(baseline, baseline_confirm_args)
        fused_confirm = _benchmark_us(fused, fused_confirm_args)
        del baseline_confirm_args
        del fused_confirm_args
        torch.cuda.empty_cache()

        average_baseline = (baseline_first + baseline_confirm) / 2.0
        average_fused = (fused_first + fused_confirm) / 2.0
        results["first_sweep_us"][run_name] = {
            "baseline": baseline_first,
            "fused": fused_first,
        }
        results["confirm_sweep_us"][run_name] = {
            "baseline": baseline_confirm,
            "fused": fused_confirm,
        }
        results["average_us"][run_name] = {
            "baseline": average_baseline,
            "fused": average_fused,
        }
        results["average_speedup_ratio_vs_composed_bridge_baseline"][
            run_name
        ] = (average_baseline / average_fused)
        del baseline
        del fused
        torch.cuda.empty_cache()

    ratios = list(
        results["average_speedup_ratio_vs_composed_bridge_baseline"].values()
    )
    large_ratios = [
        results["average_speedup_ratio_vs_composed_bridge_baseline"][name]
        for name in large_shape_names
    ]
    confirm_ratios = [
        results["confirm_sweep_us"][name]["baseline"]
        / results["confirm_sweep_us"][name]["fused"]
        for name in results["confirm_sweep_us"]
    ]
    confirm_large_ratios = [
        results["confirm_sweep_us"][name]["baseline"]
        / results["confirm_sweep_us"][name]["fused"]
        for name in large_shape_names
    ]

    results["average_geomean_speedup_vs_composed_bridge_baseline"] = (
        _geomean(ratios)
    )
    results[
        "average_large_shape_geomean_speedup_vs_composed_bridge_baseline"
    ] = _geomean(large_ratios)
    results["confirm_geomean_speedup_vs_composed_bridge_baseline"] = _geomean(
        confirm_ratios
    )
    results[
        "confirm_large_shape_geomean_speedup_vs_composed_bridge_baseline"
    ] = _geomean(confirm_large_ratios)
    results["localization"] = (
        "composed_bridge_passes_when_attn_out_is_observed_"
        "suggesting_an_unobserved_graph_seam_issue"
        if observe_attn_out
        else "composed_bridge_output_only_passes_without_attn_out_observation"
    )
    _print_results(results)
