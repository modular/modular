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
"""HuggingFace -> MAX weight adapter for the Mamba2 SSD architecture.

Mirrors :mod:`max.pipelines.architectures.mamba.weight_adapters` for the
Mamba2 mixer layout. The adapter consumes a :class:`Weights` accessor
(the same one the safetensor loader produces in the pipeline runtime)
and emits a ``dict[str, WeightData]`` keyed by the MAX weight names
expected by :class:`Mamba2Block` / :class:`Mamba2Mixer`.

Two non-trivial transformations live here:

1. **in_proj.weight slice.** HF stores the fused ``in_proj`` weight as
   ``[projection_size, d_model]`` where ``projection_size = 2*d_mlp +
   d_inner + (d_inner + 2*ngroups*d_state) + nheads`` and the row
   ordering is ``[d_mlp, d_mlp, gate, hidden_states_B_C, dt]`` (see
   ``transformers.models.mamba2.modeling_mamba2``'s split call). Our
   :class:`Mamba2Mixer` was built with ``d_ssm == d_inner`` so it
   expects ``[z, xBC, dt]`` (3 slots, no MLP). The adapter drops the
   leading ``2 * d_mlp`` rows. With the state-spaces/mamba2 reference
   checkpoints ``d_mlp == 0`` and the slice is a no-op; we still issue
   the slice unconditionally so adapter behaviour is the same on every
   checkpoint.
2. **conv1d.weight squeeze.** HF stores depthwise conv weights as
   ``[conv_dim, 1, d_conv]``; our NN module's ``conv1d_weight`` is
   ``[conv_dim, d_conv]``. Mirrors the Mamba1 adapter.

Other tensors (``A_log``, ``D``, ``dt_bias``, ``embeddings.weight``,
``norm_f.weight``, ``out_proj.weight``, per-layer ``norm.weight``) only
need the prefix/name rename pass and — for ``A_log`` / ``dt_bias`` —
an explicit ``astype(float32)``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib.config.model_config import _select_dtype_cast
from max.pipelines.modeling.config_enums import supported_encoding_dtype

from .model_config import Mamba2Config

logger = logging.getLogger("max.pipelines.mamba2")


# Ordered rename map; mirrors `MAMBA_SAFETENSOR_MAPPING`. Replacements
# are applied in order — `backbone.` strip must run before `model.` for
# any future repo that nests both.
MAMBA2_SAFETENSOR_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
        ("backbone.", ""),
        ("model.", ""),
        # HF "embeddings" (plural) -> MAX "embedding" (singular).
        ("embeddings.weight", "embedding.weight"),
        # HF final norm `norm_f` -> MAX `norm`.
        ("norm_f.weight", "norm.weight"),
        # HF dotted depthwise conv params -> MAX underscored attribute
        # names (Mamba2Mixer stores `conv1d_weight` / `conv1d_bias`
        # rather than a sub-Module with a `weight` / `bias` field).
        ("conv1d.weight", "conv1d_weight"),
        ("conv1d.bias", "conv1d_bias"),
    ]
)


def _rename(name: str) -> str:
    """Apply :data:`MAMBA2_SAFETENSOR_MAPPING` to ``name`` in order."""
    out = name
    for before, after in MAMBA2_SAFETENSOR_MAPPING.items():
        out = out.replace(before, after)
    return out


def _as_numpy(wd: WeightData) -> np.ndarray:
    """Materialize a :class:`WeightData` as a contiguous numpy array.

    Uses the DLPack protocol so this works regardless of whether the
    underlying storage is a numpy view, a torch tensor, or a safetensor
    buffer. The ``.copy()`` is defensive — slicing in numpy returns a
    view, and downstream consumers expect a standalone buffer.
    """
    return np.from_dlpack(wd).copy()


def _slice_in_proj(wd: WeightData, config: Mamba2Config) -> WeightData:
    """Drop the leading ``2*d_mlp`` rows from an HF ``in_proj.weight``.

    HF row ordering: ``[d_mlp, d_mlp, gate, hidden_states_B_C, dt]`` with
    sizes ``[d_mlp, d_mlp, d_inner, conv_dim, nheads]``. Our NN module
    expects ``[z, xBC, dt]`` (no MLP, ``d_ssm == d_inner``).

    The slice is taken along axis 0 (the ``projection_size`` axis); axis
    1 is ``d_model`` and unchanged. Concatenation isn't needed because
    the surviving three slots are already contiguous in HF's layout.

    With ``d_mlp == 0`` (the only configuration the state-spaces/mamba2
    family of repos ships) this is a no-op and we keep the original
    buffer. With ``d_mlp > 0`` we slice and rewrap.
    """
    shape = tuple(int(d) for d in wd.shape)
    if len(shape) != 2:
        raise ValueError(f"in_proj.weight expected 2D, got shape {shape}")
    projection_size, d_model = shape
    if d_model != config.d_model:
        raise ValueError(
            f"in_proj.weight axis-1 ({d_model}) does not match config "
            f"d_model ({config.d_model})"
        )

    # HF's projection_size when `d_ssm == d_inner`:
    #   projection_size = 2 * d_mlp + d_in_proj_hf
    # where ``d_in_proj_hf`` is the HF in_proj row count BEFORE the
    # B/C broadcast (i.e. computed with the source checkpoint's
    # ``n_groups``, not our post-broadcast ``config.ngroups``).
    # ``config.source_ngroups`` is populated by ``from_hf_config`` for
    # this purpose; non-HF callers (``ngroups`` already final) fall back
    # to ``config.d_in_proj``.
    if (
        config.source_ngroups is not None
        and config.source_ngroups != config.ngroups
    ):
        expected_d_in_proj = (
            2 * config.d_inner
            + 2 * config.source_ngroups * config.d_state
            + config.nheads
        )
    else:
        expected_d_in_proj = config.d_in_proj
    extra = projection_size - expected_d_in_proj
    if extra < 0 or extra % 2 != 0:
        raise ValueError(
            f"in_proj.weight projection_size={projection_size} is not "
            f"compatible with config d_in_proj={expected_d_in_proj}; "
            "expected a non-negative even surplus for the d_mlp slot."
        )
    d_mlp = extra // 2

    if d_mlp == 0:
        return wd

    logger.info(
        f"Mamba2 adapter: dropping {2 * d_mlp} d_mlp rows from "
        f"in_proj.weight (projection_size={projection_size}, "
        f"d_in_proj={expected_d_in_proj})"
    )

    arr = _as_numpy(wd)
    # Drop the first `2 * d_mlp` rows; surviving order is
    # [z (d_inner), xBC (HF conv_dim), dt (nheads)], which matches what
    # ``Mamba2Mixer.in_proj`` produces under the HF group layout. When
    # ``source_ngroups < nheads`` the surviving rows total
    # ``2*d_inner + 2*source_ngroups*d_state + nheads``; the subsequent
    # ``_broadcast_in_proj_BC`` step tiles B/C up to ``nheads``.
    sliced = arr[2 * d_mlp :, :]
    src_ngroups = (
        config.source_ngroups
        if config.source_ngroups is not None
        else config.ngroups
    )
    expected_rows = (
        2 * config.d_inner + 2 * src_ngroups * config.d_state + config.nheads
    )
    if sliced.shape[0] != expected_rows:
        raise ValueError(
            f"in_proj.weight slice produced shape {sliced.shape}, "
            f"expected ({expected_rows}, {config.d_model})"
        )
    return WeightData.from_numpy(sliced.copy(), wd.name)


def _broadcast_in_proj_BC(wd: WeightData, config: Mamba2Config) -> WeightData:
    """If ``ngroups < nheads``, tile the ``B`` / ``C`` slots of the
    already-sliced ``in_proj.weight`` so the kernel ABI's
    ``ngroups == nheads`` assumption holds.

    Row ordering after :func:`_slice_in_proj`:
    ``[z (d_inner), x (d_inner), B (ngroups*d_state), C (ngroups*d_state), dt (nheads)]``
    where ``z + x`` together form the gate + ``x``-projection slots and
    are independent of ``ngroups``. We tile only ``B`` and ``C``.

    Note: this isn't reachable from `Mamba2Mixer` directly (item 3
    forces ``ngroups == nheads`` at construction time), but the adapter
    needs to handle it because HF's ``transformers.Mamba2Config`` ships
    with ``n_groups=8`` while ``num_heads=128`` — the only way to load
    a real upstream checkpoint into our mixer is to broadcast at the
    weight-loading stage.
    """
    # ``config.ngroups`` is the *post-broadcast* group count (forced to
    # ``nheads`` for HF checkpoints by ``from_hf_config``). Read the source
    # group count from ``config.source_ngroups`` so we can detect whether
    # broadcasting is actually needed.
    src_ngroups = (
        config.source_ngroups
        if config.source_ngroups is not None
        else config.ngroups
    )
    if src_ngroups == config.nheads:
        return wd
    if config.nheads % src_ngroups != 0:
        raise ValueError(
            f"source_ngroups ({src_ngroups}) must divide nheads "
            f"({config.nheads}) to broadcast B/C in in_proj.weight."
        )
    tile = config.nheads // src_ngroups

    arr = _as_numpy(wd)
    if arr.ndim != 2 or arr.shape[1] != config.d_model:
        raise ValueError(
            f"in_proj.weight slice has unexpected shape {arr.shape}; "
            f"expected (*, {config.d_model})."
        )

    # Slice along axis 0 in the canonical order produced by
    # `_slice_in_proj`: [z, x, B, C, dt].
    d_inner = config.d_inner
    nheads = config.nheads
    d_state = config.d_state
    # `z` and `x` live in the gate + xBC slots from our Mixer's POV.
    # After the slice, the rows are laid out as:
    #   [z: d_inner, x: d_inner, B: src_ngroups*d_state, C: src_ngroups*d_state, dt: nheads]
    # When ``src_ngroups < nheads`` we tile B and C along the group axis.
    src_bc_block = src_ngroups * d_state
    offsets = [
        0,
        d_inner,  # end of z
        2 * d_inner,  # end of x
        2 * d_inner + src_bc_block,  # end of B
        2 * d_inner + 2 * src_bc_block,  # end of C
        2 * d_inner + 2 * src_bc_block + nheads,  # end of dt
    ]
    if arr.shape[0] != offsets[-1]:
        raise ValueError(
            f"in_proj.weight slice rows={arr.shape[0]} does not match "
            f"expected {offsets[-1]} for tiling."
        )

    z = arr[offsets[0] : offsets[1], :]
    x = arr[offsets[1] : offsets[2], :]
    B = arr[offsets[2] : offsets[3], :]
    C = arr[offsets[3] : offsets[4], :]
    dt = arr[offsets[4] : offsets[5], :]

    # Reshape B / C to (src_ngroups, d_state, d_model), tile along the
    # group axis to nheads, then flatten back.
    B_g = B.reshape(src_ngroups, d_state, config.d_model)
    C_g = C.reshape(src_ngroups, d_state, config.d_model)
    B_tiled = np.repeat(B_g, tile, axis=0).reshape(
        nheads * d_state, config.d_model
    )
    C_tiled = np.repeat(C_g, tile, axis=0).reshape(
        nheads * d_state, config.d_model
    )

    new = np.concatenate([z, x, B_tiled, C_tiled, dt], axis=0)
    expected_rows = 2 * d_inner + 2 * nheads * d_state + nheads
    if new.shape[0] != expected_rows:
        raise ValueError(
            f"in_proj.weight tile produced shape {new.shape}, "
            f"expected ({expected_rows}, {config.d_model})"
        )
    return WeightData.from_numpy(new.copy(), wd.name)


def _squeeze_conv1d_weight(wd: WeightData) -> WeightData:
    """Reshape ``[conv_dim, 1, d_conv]`` -> ``[conv_dim, d_conv]``.

    HF stores depthwise conv1d weights with an explicit
    in-channels-per-group axis of size 1; our :class:`Mamba2Mixer`
    stores it as a plain 2D matrix.
    """
    shape = tuple(int(d) for d in wd.shape)
    if len(shape) == 3 and shape[1] == 1:
        arr = _as_numpy(wd)
        arr = arr.reshape(shape[0], shape[2]).copy()
        return WeightData.from_numpy(arr, wd.name)
    if len(shape) == 2:
        # Already squeezed (e.g. converted by an upstream tool); pass
        # through unchanged.
        return wd
    raise ValueError(
        f"conv1d.weight expected shape [conv_dim, 1, d_conv] or "
        f"[conv_dim, d_conv], got {shape}"
    )


def _broadcast_conv1d_BC(
    wd: WeightData, config: Mamba2Config, axis0_is_bias: bool = False
) -> WeightData:
    """Tile the B and C channel slots of a conv1d weight or bias.

    HF stores the depthwise conv over ``conv_dim_hf == d_inner + 2 *
    source_ngroups * d_state`` channels. Our mixer's conv operates over
    ``conv_dim == d_inner + 2 * nheads * d_state`` (the post-broadcast
    group count, matching the SSD kernel ABI). When ``source_ngroups <
    nheads`` we tile the per-channel B and C slots so each per-head B/C
    column gets its own conv filter (replicated from the source group).

    For ``axis0_is_bias=False`` (weight): shape is ``(conv_dim, d_conv)``.
    For ``axis0_is_bias=True`` (bias): shape is ``(conv_dim,)``.
    Tiling happens along the channel axis (axis 0).
    """
    src_ngroups = (
        config.source_ngroups
        if config.source_ngroups is not None
        else config.ngroups
    )
    if src_ngroups == config.nheads:
        return wd
    if config.nheads % src_ngroups != 0:
        raise ValueError(
            f"source_ngroups ({src_ngroups}) must divide nheads "
            f"({config.nheads}) to broadcast conv1d B/C."
        )

    arr = _as_numpy(wd)
    d_inner = config.d_inner
    d_state = config.d_state
    nheads = config.nheads
    tile = nheads // src_ngroups
    src_bc = src_ngroups * d_state

    expected_axis0 = d_inner + 2 * src_bc
    if arr.shape[0] != expected_axis0:
        raise ValueError(
            f"conv1d weight/bias axis-0 ({arr.shape[0]}) does not match "
            f"expected pre-broadcast {expected_axis0} "
            f"(d_inner={d_inner}, src_ngroups={src_ngroups}, d_state={d_state})."
        )

    if axis0_is_bias:
        x_part = arr[:d_inner]
        B_part = arr[d_inner : d_inner + src_bc]
        C_part = arr[d_inner + src_bc : d_inner + 2 * src_bc]
        B_g: np.ndarray = B_part.reshape(src_ngroups, d_state)
        C_g: np.ndarray = C_part.reshape(src_ngroups, d_state)
        B_tiled: np.ndarray = np.repeat(B_g, tile, axis=0).reshape(
            nheads * d_state
        )
        C_tiled: np.ndarray = np.repeat(C_g, tile, axis=0).reshape(
            nheads * d_state
        )
        new = np.concatenate([x_part, B_tiled, C_tiled], axis=0)
    else:
        d_conv = arr.shape[1]
        x_part = arr[:d_inner, :]
        B_part = arr[d_inner : d_inner + src_bc, :]
        C_part = arr[d_inner + src_bc : d_inner + 2 * src_bc, :]
        B_g = B_part.reshape(src_ngroups, d_state, d_conv)
        C_g = C_part.reshape(src_ngroups, d_state, d_conv)
        B_tiled = np.repeat(B_g, tile, axis=0).reshape(nheads * d_state, d_conv)
        C_tiled = np.repeat(C_g, tile, axis=0).reshape(nheads * d_state, d_conv)
        new = np.concatenate([x_part, B_tiled, C_tiled], axis=0)

    return WeightData.from_numpy(new.copy(), wd.name)


def convert_mamba2_state_dict(
    weights: Weights | dict[str, Weights],
    config: Mamba2Config | None = None,
    *,
    huggingface_config: Any = None,
    pipeline_config: Any = None,
    **_unused: Any,
) -> dict[str, WeightData]:
    """Translate an HF Mamba2 state dict into the MAX weight namespace.

    The pipeline registry invokes adapters with
    ``adapter(weights_dict, huggingface_config=..., pipeline_config=...)``.
    Standalone test/script callers pass a :class:`Mamba2Config` via
    ``config`` directly. We accept either form: if ``config`` is omitted
    we build it from ``huggingface_config``.

    Args:
        weights: The :class:`Weights` accessor produced by the
            safetensor loader (or any mapping from weight name to
            :class:`Weights`). A plain ``dict[str, Weights]`` is
            accepted for ease of testing.
        config: The :class:`Mamba2Config` describing the target shape.
            If ``None``, derived from ``huggingface_config``.
        huggingface_config: The HF ``AutoConfig`` (pipeline-supplied).
        pipeline_config: The MAX ``PipelineConfig`` (unused by this
            adapter, but accepted to satisfy the registry's call shape).

    Returns:
        Dictionary mapping MAX weight names to :class:`WeightData`. Use
        this directly as the ``state_dict`` argument when calling
        ``Mamba2Block.load_state_dict(...)`` / equivalent.
    """
    if config is None:
        if huggingface_config is None:
            raise ValueError(
                "convert_mamba2_state_dict requires either `config` or "
                "`huggingface_config` to be provided."
            )
        config = Mamba2Config.from_hf_config(huggingface_config)
    # Normalize to an iterator of (name, Weights) pairs.
    if isinstance(weights, dict):
        items = list(weights.items())
    else:
        items = list(weights.items())

    new_state_dict: dict[str, WeightData] = {}

    for hf_name, weight in items:
        max_name = _rename(hf_name)
        wd = weight.data()

        # in_proj weight needs the d_mlp slice (axis-0 drop), then a
        # B/C broadcast when `ngroups < nheads`. Catch both `in_proj.weight`
        # and any quantized variant (`in_proj.scales` / `.qweight`) by
        # keying on the *.in_proj. prefix — but only mutate `.weight`
        # since that's the only tensor with the per-row block layout.
        # TODO(verify-vs-hf): if a real INT-quantized checkpoint surfaces,
        # the scales tensor will need the same row-slice. Cross that
        # bridge in the integration test.
        if max_name.endswith("in_proj.weight"):
            wd = _slice_in_proj(wd, config)
            wd = _broadcast_in_proj_BC(wd, config)

        elif "conv1d_weight" in max_name:
            wd = _squeeze_conv1d_weight(wd)
            wd = _broadcast_conv1d_BC(wd, config, axis0_is_bias=False)

        elif "conv1d_bias" in max_name:
            wd = _broadcast_conv1d_BC(wd, config, axis0_is_bias=True)

        # Force fp32 for A_log and dt_bias — both feed math paths
        # (the `-exp(A_log)` in the mixer and the `softplus(dt + dt_bias)`
        # inside the SSD kernel) that lose accuracy in lower precision.
        # The reference module declares both as fp32 parameters; HF
        # checkpoints sometimes ship them as bf16/fp16.
        elif max_name.endswith(("A_log", "dt_bias")):
            if wd.dtype != DType.float32:
                wd = wd.astype(DType.float32)

        new_state_dict[max_name] = wd

    # Apply the pipeline's resolved load-time dtype cast (main casts
    # float32 checkpoints to bfloat16 on GPU) so the loaded tensors match
    # the dtype the NN modules are built with under ``default_dtype``.
    # Mirrors Mamba1's adapter. The SSD kernels are dtype-generic and do
    # their accumulation in fp32 internally, so a bf16 ``A_log`` /
    # ``dt_bias`` only costs the storage precision of the parameters.
    # TODO(MXF-517): this should be resolved by the ArchConfig, not the adapter.
    if pipeline_config is not None:
        cast_from, cast_to = _select_dtype_cast(
            pipeline_config.model, "float32"
        )
        if cast_from:
            assert cast_to, (
                "Invalid configuration: applied_dtype_cast_to is not set "
                "but applied_dtype_cast_from is set. This should not happen."
            )
            from_dtype = supported_encoding_dtype(cast_from)
            to_dtype = supported_encoding_dtype(cast_to)
            for key, wd in new_state_dict.items():
                if wd.dtype == from_dtype:
                    new_state_dict[key] = wd.astype(to_dtype)

    return new_state_dict


__all__ = [
    "MAMBA2_SAFETENSOR_MAPPING",
    "convert_mamba2_state_dict",
]


# ---------------------------------------------------------------------------
# Smoke test (no real checkpoint required).
#
# Run via:
#   python -m max.pipelines.architectures.mamba2.weight_adapters
#
# Exercises the dataclass + adapter on a synthetic state dict so a future
# refactor of the slicing math gets an immediate signal. Loading a real
# HF checkpoint is blocked at runtime by the same `state_space.mojoc`
# Tuple-shape bug that blocks the NN modules, so this `__main__` block
# is the closest thing to a unit test that fits in-scope.
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Construct a fake HF state dict and run the adapter end-to-end."""
    import dataclasses

    # Use a small config so the synthetic tensors are tiny.
    cfg = Mamba2Config(
        d_model=128,
        n_layer=2,
        vocab_size=256,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        # `ngroups == nheads` to exercise the no-broadcast path; the
        # broadcast path has its own assertion at construction time.
        ngroups=8,
        chunk_size=64,
    )
    assert cfg.d_inner == 256
    assert cfg.nheads == 8
    assert cfg.ngroups == cfg.nheads
    assert cfg.d_in_proj == 2 * 256 + 2 * 8 * 16 + 8 == 776
    assert cfg.conv_dim == 256 + 2 * 8 * 16 == 512

    # Cover the HF d_mlp=0 path: projection_size == d_in_proj.
    in_proj_hf = np.random.randn(cfg.d_in_proj, cfg.d_model).astype(np.float32)
    # Cover conv1d: HF shape [conv_dim, 1, d_conv].
    conv1d_w_hf = np.random.randn(cfg.conv_dim, 1, cfg.d_conv).astype(
        np.float32
    )
    conv1d_b_hf = np.random.randn(cfg.conv_dim).astype(np.float32)
    A_log_hf = np.random.randn(cfg.nheads).astype(np.float16)  # bf16-ish
    dt_bias_hf = np.random.randn(cfg.nheads).astype(np.float16)
    D_hf = np.random.randn(cfg.nheads).astype(np.float32)
    embed_hf = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32)
    norm_w_hf = np.random.randn(cfg.d_model).astype(np.float32)
    out_proj_hf = np.random.randn(cfg.d_model, cfg.d_inner).astype(np.float32)
    layer_norm_w_hf = np.random.randn(cfg.d_model).astype(np.float32)

    # Fake `Weights`-like wrapper: just needs `.data()` -> WeightData.
    @dataclasses.dataclass
    class _FakeWeight:
        wd: WeightData

        def data(self) -> WeightData:
            return self.wd

    def _w(arr: np.ndarray, name: str) -> _FakeWeight:
        return _FakeWeight(WeightData.from_numpy(arr, name))

    fake_state: dict[str, _FakeWeight] = {
        "backbone.embeddings.weight": _w(embed_hf, "embed"),
        "backbone.norm_f.weight": _w(norm_w_hf, "norm_f"),
        "backbone.layers.0.norm.weight": _w(layer_norm_w_hf, "norm0"),
        "backbone.layers.0.mixer.in_proj.weight": _w(in_proj_hf, "in_proj0"),
        "backbone.layers.0.mixer.conv1d.weight": _w(conv1d_w_hf, "conv1d_w0"),
        "backbone.layers.0.mixer.conv1d.bias": _w(conv1d_b_hf, "conv1d_b0"),
        "backbone.layers.0.mixer.A_log": _w(A_log_hf, "A_log0"),
        "backbone.layers.0.mixer.dt_bias": _w(dt_bias_hf, "dt_bias0"),
        "backbone.layers.0.mixer.D": _w(D_hf, "D0"),
        "backbone.layers.0.mixer.out_proj.weight": _w(out_proj_hf, "out_proj0"),
    }

    out = convert_mamba2_state_dict(fake_state, cfg)  # type: ignore[arg-type]

    # Rename checks.
    assert "embedding.weight" in out
    assert "norm.weight" in out
    assert "layers.0.mixer.in_proj.weight" in out
    assert "layers.0.mixer.conv1d_weight" in out
    assert "layers.0.mixer.conv1d_bias" in out
    assert "layers.0.mixer.A_log" in out
    assert "layers.0.mixer.dt_bias" in out

    # Shape checks.
    assert tuple(
        int(d) for d in out["layers.0.mixer.in_proj.weight"].shape
    ) == (
        cfg.d_in_proj,
        cfg.d_model,
    )
    assert tuple(int(d) for d in out["layers.0.mixer.conv1d_weight"].shape) == (
        cfg.conv_dim,
        cfg.d_conv,
    )

    # dtype promotions.
    assert out["layers.0.mixer.A_log"].dtype == DType.float32
    assert out["layers.0.mixer.dt_bias"].dtype == DType.float32

    # Now exercise the d_mlp > 0 path: prepend 2*d_mlp rows to in_proj
    # and re-run. The adapter should slice them off.
    d_mlp = 4
    in_proj_hf_padded = np.concatenate(
        [
            np.random.randn(2 * d_mlp, cfg.d_model).astype(np.float32),
            in_proj_hf,
        ],
        axis=0,
    )
    fake_state["backbone.layers.0.mixer.in_proj.weight"] = _w(
        in_proj_hf_padded, "in_proj_padded"
    )
    out2 = convert_mamba2_state_dict(fake_state, cfg)  # type: ignore[arg-type]
    assert tuple(
        int(d) for d in out2["layers.0.mixer.in_proj.weight"].shape
    ) == (
        cfg.d_in_proj,
        cfg.d_model,
    )

    # And the broadcast path: a config where ngroups < nheads.
    cfg_bcast = Mamba2Config(
        d_model=128,
        n_layer=2,
        vocab_size=256,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=2,  # nheads=8, tile=4
        chunk_size=64,
    )
    # HF's projection_size for ngroups=2 (smaller than the ngroups=8
    # config above) — re-derive locally rather than reusing cfg.d_in_proj.
    in_proj_bcast = np.random.randn(
        cfg_bcast.d_in_proj, cfg_bcast.d_model
    ).astype(np.float32)
    fake_state_bcast = {
        "backbone.layers.0.mixer.in_proj.weight": _w(
            in_proj_bcast, "in_proj_bcast"
        ),
    }
    out_bcast = convert_mamba2_state_dict(fake_state_bcast, cfg_bcast)  # type: ignore[arg-type]
    # Post-broadcast rows = 2*d_inner + 2*nheads*d_state + nheads.
    expected_rows = (
        2 * cfg_bcast.d_inner
        + 2 * cfg_bcast.nheads * cfg_bcast.d_state
        + cfg_bcast.nheads
    )
    assert tuple(
        int(d) for d in out_bcast["layers.0.mixer.in_proj.weight"].shape
    ) == (
        expected_rows,
        cfg_bcast.d_model,
    )

    print("mamba2 weight_adapters smoke: OK")


if __name__ == "__main__":
    _smoke()
