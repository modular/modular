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
"""The Mistral Large 4 weight contract: checkpoint name to module name.

The NVFP4 checkpoint is a ModelOpt mixed-precision export in Mistral's
consolidated naming (no ``model.`` prefix). Mach's loader renames its tensors
through the registry's prefix/replacement table and graft, while MAX's loader
renames them in a Python adapter; if the two disagree by one tensor, Mach hands
the compiled graph a weight under a name it never asks for and MAX reads
whatever the MEF's dtype width says from a pointer that does not hold it. This
module is the single statement of the mapping and
``engine_core::mistral_large4`` is its Rust mirror: this module emits the table
as :meth:`ContractTable.to_json`, which is the committed golden, and the
contract test asserts that Mach's table equals it tensor for tensor.

Two ModelOpt scale conventions share the ``.weight_scale`` name in this
checkpoint, and the sibling rule below is what tells them apart:

* ``FP8_PB_WO`` modules (``wq``, ``wo``, ``indexer.wq_b``, the shared experts,
  layer 0's dense MLP) ship ``weight_scale`` as a **4-D** ``float32``
  ``[rows/128, 1, cols/128, 1]``. ``FP8BlockTensor`` wants the 2-D block grid,
  so the two unit axes are squeezed and the tensor is keyed
  ``weight_scale_inv``, which is the name the graph declares.
* ``NVFP4`` modules (the routed experts) ship ``weight_scale`` as an already
  2-D ``float8_e4m3fn`` per-16-element grid, beside ``weight_scale_2`` and
  ``input_scale``. They keep all three names and their shapes.

The squeeze is a reshape, not a byte transform: the 4-D tensor holds exactly
the 2-D grid's elements in the same order, so neither loader owns a copy of a
scale's bytes. ModelOpt's per-tensor globals are likewise already the
convention MAX's NVFP4 matmul multiplies by, so nothing is inverted here.
(``engine_safetensors::adapt``'s compressed-tensors rule inverts *its* globals;
applying that inversion to a ModelOpt export would scale every expert it
dequantizes by ``global^2``.)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

# Storage dtype spellings as the checkpoint's safetensors headers write them.
DTYPE_BF16 = "BF16"
DTYPE_F32 = "F32"
DTYPE_F8_E4M3 = "F8_E4M3"
DTYPE_U8 = "U8"

# The two quantization conventions this checkpoint family uses. ModelOpt's
# ``FP8_PB_WO`` blocks a weight per 128x128 tile with no activation scale
# (activations are quantized dynamically); ``NVFP4`` packs two E2M1 values per
# byte with a per-16-element E4M3 scale plus two per-tensor ``float32`` scales.
QUANT_ALGO_FP8_PB_WO = "FP8_PB_WO"
QUANT_ALGO_NVFP4 = "NVFP4"

# Per-tensor scale suffixes; their presence decides a module's convention.
SCALE_SUFFIX = ".weight_scale"
NVFP4_GLOBAL_SUFFIX = ".weight_scale_2"
NVFP4_INPUT_SUFFIX = ".input_scale"
# The base FP8 checkpoint's 2-D block scales, which use a different convention
# from either of the above and are not wired yet.
BASE_REPO_SCALE_SUFFIX = ".qscale_weight"

# Roles, spelled identically in the Rust mirror.
ROLE_PLAIN = "plain"
ROLE_FP8_WEIGHT = "fp8_weight"
ROLE_FP8_SCALE = "fp8_scale"
ROLE_NVFP4_WEIGHT = "nvfp4_weight"
ROLE_NVFP4_GROUP_SCALE = "nvfp4_group_scale"
ROLE_NVFP4_GLOBAL_SCALE = "nvfp4_global_scale"
ROLE_NVFP4_INPUT_SCALE = "nvfp4_input_scale"
ROLE_VISION_QK = "vision_qk_rope"

# The name the graph declares for an FP8 block-scaled weight's scales.
FP8_SCALE_NAME_SUFFIX = ".weight_scale_inv"

# The rename table, applied exactly as Mach's registry applies it: strip the
# first matching prefix, then each replacement in order to every occurrence,
# then graft the module tree's root.
WEIGHT_NAME_PREFIXES: tuple[str, ...] = ()
WEIGHT_NAME_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # `attention_norm` is not attention. The two patterns cannot interact:
    # `.attention.` requires a literal dot where `.attention_norm.` has the
    # underscore, so this rule's position in the table is a readability choice.
    (".attention_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    (".attention.", ".self_attn."),
    (".gate.weight", ".mlp.gate.gate_score.weight"),
    (".feed_forward.", ".mlp."),
    (".experts.", ".mlp.experts."),
    (".shared_experts.", ".mlp.shared_experts."),
    # The untied head. No `mtp_dspark.` name reaches this table.
    ("output.weight", "lm_head.weight"),
)
WEIGHT_NAME_GRAFT = "language_model."

# Checkpoint subtrees the language graph does not hold, with the reason and
# the module that binds them instead. An unlisted unknown name fails here
# rather than loading silently under a name nothing reads.
_SKIPPED_PREFIX_REASONS: tuple[tuple[str, str, str], ...] = (
    ("mtp_dspark.", "dspark", "Mistral Large 4 DSpark drafter"),
    (
        "vision_encoder.",
        "vision",
        "Pixtral vision tower, the vision graph holds it",
    ),
    (
        "pre_mm_projector_norm.",
        "vision",
        "Pixtral vision projector norm, the vision graph holds it",
    ),
    (
        "patch_merger.",
        "vision",
        "Pixtral patch merger, the vision graph holds it",
    ),
    (
        "vision_language_adapter.",
        "vision",
        "Pixtral vision-language adapter, the vision graph holds it",
    ),
)

# The vision graph subtrees: the tower, the projector norm, the patch
# merger and the vision-language adapter. The language graph skips them by
# prefix (above); the vision graph binds them through the vision table below.
# The text replacements must never touch these names: `.attention.` ->
# `.self_attn.` and `.feed_forward.` -> `.mlp.` would rewrite the tower
# module names, so the vision route is prefix-local and identity-except-one.
VISION_PREFIXES: tuple[str, ...] = (
    "vision_encoder.",
    "pre_mm_projector_norm.",
    "patch_merger.",
    "vision_language_adapter.",
)

# The conv patch embed is a raw graph tensor, not a Linear, so its `.weight`
# suffix is the one vision rename.
_VISION_PATCH_CONV_SOURCE = "vision_encoder.patch_conv.weight"
_VISION_PATCH_CONV_NAME = "vision_encoder.patch_conv"

# The tower's Q and K projection rows are stored in the checkpoint's
# interleaved-RoPE layout -- within each head, adjacent row pairs
# ``(x0, x1), (x2, x3), ...`` rotate together -- while the tower computes
# rotate-half rope, like HF Pixtral. The adapter permutes each head's rows
# into the half-split layout, mirroring transformers' own loading rule for
# this checkpoint ("Vision model Q and K projectors: saved for interleaved
# RoPE, Pixtral expects rotate-half",
# transformers/conversion_mapping.py, mapping["mistral_large4"]).
_VISION_QK_SOURCE_PREFIX = "vision_encoder.transformer.layers."
_VISION_QK_SOURCE_SUFFIXES = (".attention.wq.weight", ".attention.wk.weight")
VISION_ATTENTION_HEADS = 16
VISION_HEAD_DIM = 128


def is_vision_name(name: str) -> bool:
    """Whether the vision graph binds ``name`` (post- or pre-adapter)."""
    return name.startswith(VISION_PREFIXES)


def skip_reason(name: str) -> str | None:
    """Returns why the language graph does not load ``name``.

    Args:
        name: A checkpoint tensor name.

    Returns:
        ``"dspark"``, ``"vision"`` -- the vision graph binds those, not
        the language graph -- or ``None`` when the language graph does.
    """
    for prefix, reason, _ in _SKIPPED_PREFIX_REASONS:
        if name.startswith(prefix):
            return reason
    return None


def skipped_tensor_documentation() -> tuple[tuple[str, str, str], ...]:
    """Returns the documented skip list as ``(prefix, reason, detail)``."""
    return _SKIPPED_PREFIX_REASONS


def transform_weight_name(name: str) -> str:
    """Maps a checkpoint tensor name onto its MAX module path.

    Args:
        name: A checkpoint tensor name.

    Returns:
        The module tree path the MEF declares.
    """
    stripped = name
    for prefix in WEIGHT_NAME_PREFIXES:
        if prefix and name.startswith(prefix):
            stripped = name[len(prefix) :]
            break
    for old, new in WEIGHT_NAME_REPLACEMENTS:
        stripped = stripped.replace(old, new)
    return WEIGHT_NAME_GRAFT + stripped


@dataclass(frozen=True)
class AdaptedTensor:
    """One checkpoint tensor's module name, storage dtype, shape and role.

    ``role`` says how the bytes are read: a plain tensor, an FP8 weight or the
    2-D grid of its block scales, an NVFP4 packed weight, or one of the two
    per-tensor NVFP4 scales.
    """

    source: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class ContractTable:
    """The full name/shape/dtype/scale-layout table for one checkpoint."""

    tensors: tuple[AdaptedTensor, ...]
    skipped: tuple[tuple[str, str], ...]

    def to_json(self) -> dict[str, object]:
        """Returns the table as the golden both loaders are compared against."""
        return {
            "tensors": [
                {
                    "source": tensor.source,
                    "name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "role": tensor.role,
                }
                for tensor in self.tensors
            ],
            "skipped": [
                {"source": source, "reason": reason}
                for source, reason in self.skipped
            ],
        }


def _weight_shape(
    headers: Mapping[str, tuple[str, tuple[int, ...]]], module: str
) -> tuple[int, int]:
    """Returns the 2-D shape of the weight a module's scales belong to.

    Args:
        headers: The whole checkpoint's ``name -> (dtype, shape)``.
        module: A checkpoint module path, e.g. ``layers.0.attention.wq``.

    Returns:
        The quantized weight's ``[rows, cols]``.

    Raises:
        ValueError: If the module ships no ``.weight``, or it is not 2-D.
    """
    key = f"{module}.weight"
    if key not in headers:
        raise ValueError(f"{module}: scales without a `.weight` sibling")
    shape = headers[key][1]
    if len(shape) != 2:
        raise ValueError(f"{key} is not 2-D: {shape}")
    return shape[0], shape[1]


def _squeeze_fp8_scale(
    weight: str,
    dtype: str,
    shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Checks a ModelOpt FP8 block scale and returns its 2-D grid shape.

    Args:
        weight: The module name, for error messages.
        dtype: The scale's storage dtype.
        shape: The scale's 4-D shape as stored.
        weight_shape: The quantized weight's 2-D shape.

    Returns:
        The ``[rows, cols]`` block grid.

    Raises:
        ValueError: If the scale is not an FP8-block ``float32`` 4-D grid whose
            extents match the weight's 128x128 tiling.
    """
    rows, cols = weight_shape
    if dtype != DTYPE_F32:
        raise ValueError(
            f"{weight}: FP8 block scales must be {DTYPE_F32}, not {dtype}"
        )
    if len(shape) != 4 or shape[1] != 1 or shape[3] != 1:
        raise ValueError(
            f"{weight}: ModelOpt FP8 block scales are 4-D [rows, 1, cols, 1]; "
            f"got {shape}"
        )
    squeezed = (shape[0], shape[2])
    expected = (-(-rows // 128), -(-cols // 128))
    if squeezed != expected:
        raise ValueError(
            f"{weight}: FP8 block scale grid {squeezed} does not match the "
            f"{rows}x{cols} weight tiled at 128x128, which needs {expected}"
        )
    return squeezed


def _adapt_one(
    source: str,
    dtype: str,
    shape: tuple[int, ...],
    headers: Mapping[str, tuple[str, tuple[int, ...]]],
) -> tuple[str, str, tuple[int, ...], str]:
    """Adapts one tensor's name, dtype, shape and role.

    Args:
        source: The checkpoint tensor name.
        dtype: Its storage dtype.
        shape: Its stored shape.
        headers: The whole checkpoint's ``name -> (dtype, shape)``, read for the
            sibling tensors that decide its convention.

    Returns:
        ``(module_name, dtype, shape, role)``.

    Raises:
        ValueError: If the tensor uses a convention this loader cannot read.
    """
    if source.endswith(BASE_REPO_SCALE_SUFFIX):
        raise ValueError(
            f"{source}: the base FP8 checkpoint's 2-D qscale_weight convention "
            "is a separate loader and is not wired yet"
        )
    name = transform_weight_name(source)
    if source.endswith(SCALE_SUFFIX):
        module = source[: -len(SCALE_SUFFIX)]
        if (module + NVFP4_GLOBAL_SUFFIX) in headers:
            rows, cols = _weight_shape(headers, module)
            cols *= 2
            expected = (rows, cols // 16)
            if shape != expected:
                raise ValueError(
                    f"{source}: NVFP4 group scales are [rows, cols/16] = "
                    f"{expected}; got {shape}"
                )
            return name, dtype, shape, ROLE_NVFP4_GROUP_SCALE
        rows, cols = _weight_shape(headers, module)
        return (
            name[: -len(SCALE_SUFFIX)] + FP8_SCALE_NAME_SUFFIX,
            dtype,
            _squeeze_fp8_scale(source, dtype, shape, (rows, cols)),
            ROLE_FP8_SCALE,
        )
    if source.endswith((NVFP4_GLOBAL_SUFFIX, NVFP4_INPUT_SUFFIX)):
        if dtype != DTYPE_F32 or shape:
            raise ValueError(
                f"{source}: an NVFP4 per-tensor scale is F32 []; got {dtype} "
                f"{shape}"
            )
        role = (
            ROLE_NVFP4_GLOBAL_SCALE
            if source.endswith(NVFP4_GLOBAL_SUFFIX)
            else ROLE_NVFP4_INPUT_SCALE
        )
        return name, dtype, shape, role
    if source.endswith(".weight"):
        module = source[: -len(".weight")]
        if (module + NVFP4_GLOBAL_SUFFIX) in headers:
            if dtype != DTYPE_U8 or len(shape) != 2:
                raise ValueError(
                    f"{source}: an NVFP4 weight is packed U8 2-D; got {dtype} "
                    f"{shape}"
                )
            return name, dtype, shape, ROLE_NVFP4_WEIGHT
        if (module + SCALE_SUFFIX) in headers:
            if dtype != DTYPE_F8_E4M3 or len(shape) != 2:
                raise ValueError(
                    f"{source}: an FP8 block-scaled weight is {DTYPE_F8_E4M3} "
                    f"2-D; got {dtype} {shape}"
                )
            return name, dtype, shape, ROLE_FP8_WEIGHT
    return name, dtype, shape, ROLE_PLAIN


def transform_vision_weight_name(name: str) -> str:
    """Maps a vision checkpoint tensor name onto its vision module path.

    The vision modules spell the checkpoint consolidated names, so the
    table is the identity except the conv patch embed, whose raw graph
    tensor drops the `.weight` suffix.

    Args:
        name: A checkpoint tensor name under one of ``VISION_PREFIXES``.

    Returns:
        The vision module tree path the graph declares.
    """
    if name == _VISION_PATCH_CONV_SOURCE:
        return _VISION_PATCH_CONV_NAME
    return name


def vision_weight_table(
    headers: Mapping[str, tuple[str, tuple[int, ...]]],
) -> ContractTable:
    """Builds the vision graph contract table from checkpoint headers.

    The checkpoint leaves the whole vision subtree unquantized BF16 (the
    ModelOpt ``quantized_layers`` list omits it), so every vision tensor is
    a plain BF16 weight: a scale suffix, a packed dtype or any other
    quantized convention here is a checkpoint this loader cannot read.
    Reads names, dtypes and shapes only; no tensor data is touched. The
    tower's Q and K projections carry the ``vision_qk_rope`` role: the
    adapter permutes their rows from the checkpoint's interleaved-RoPE
    layout into the rotate-half layout the tower computes with.

    Args:
        headers: ``name -> (dtype, shape)`` over the checkpoint tensors.

    Returns:
        The vision tensors, under the vision module names.

    Raises:
        ValueError: If a vision tensor is not BF16, or two checkpoint
            tensors map onto one vision module name.
    """
    tensors: list[AdaptedTensor] = []
    for source, (dtype, shape) in headers.items():
        if not is_vision_name(source):
            continue
        if dtype != DTYPE_BF16:
            raise ValueError(
                f"{source}: the vision subtree is unquantized {DTYPE_BF16}; "
                f"got {dtype}. A quantized vision tower needs its own loader"
            )
        role = ROLE_PLAIN
        if source.startswith(_VISION_QK_SOURCE_PREFIX) and source.endswith(
            _VISION_QK_SOURCE_SUFFIXES
        ):
            if len(shape) != 2 or shape[0] != VISION_ATTENTION_HEADS * (
                VISION_HEAD_DIM
            ):
                raise ValueError(
                    f"{source}: the tower's Q/K projections must be 2-D "
                    f"with {VISION_ATTENTION_HEADS * VISION_HEAD_DIM} rows "
                    f"for the rope-layout permutation; got {dtype} {shape}"
                )
            role = ROLE_VISION_QK
        tensors.append(
            AdaptedTensor(
                source=source,
                name=transform_vision_weight_name(source),
                dtype=dtype,
                shape=shape,
                role=role,
            )
        )
    duplicates = sorted(
        name
        for name, count in Counter(tensor.name for tensor in tensors).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"vision module names collide: {duplicates[:5]}")
    return ContractTable(tensors=tuple(tensors), skipped=())


def weight_table(
    headers: Mapping[str, tuple[str, tuple[int, ...]]],
) -> ContractTable:
    """Builds the contract table from a checkpoint's safetensors headers.

    Reads names, dtypes and shapes only; no tensor data is touched.

    Args:
        headers: ``name -> (dtype, shape)`` over the checkpoint's tensors.

    Returns:
        The adapted tensors and the tensors deliberately skipped.

    Raises:
        ValueError: If a tensor uses a convention this loader cannot read, or
            two checkpoint tensors map onto one module name.
    """
    tensors: list[AdaptedTensor] = []
    skipped: list[tuple[str, str]] = []
    for source, (dtype, shape) in headers.items():
        reason = skip_reason(source)
        if reason is not None:
            skipped.append((source, reason))
            continue
        name, dtype, shape, role = _adapt_one(source, dtype, shape, headers)
        tensors.append(
            AdaptedTensor(
                source=source, name=name, dtype=dtype, shape=shape, role=role
            )
        )
    duplicates = sorted(
        name
        for name, count in Counter(tensor.name for tensor in tensors).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"module names collide: {duplicates[:5]}")
    return ContractTable(tensors=tuple(tensors), skipped=tuple(skipped))
