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
"""Functional-op wrappers for the Mamba2 SSD chunk-scan kernels.

The Mojo side registers the fused chunk-scan op under the name
``ssd_chunk_scan_combined`` in
``max/kernels/src/state_space/ssd_chunk_combined_ops.mojo``:

    @compiler.register("ssd_chunk_scan_combined")
    struct SSDChunkScanCombined[chunk_size: Int = 256]: ...

This wrapper builds the corresponding ``ops.custom`` graph node so the
op can be composed inside a ``Graph(...)`` block. The wrapper mirrors
the style of ``architectures.mamba.functional_ops.selective_scan_fwd``
(``delta_softplus: Bool`` is forwarded via ``parameters={...}``), with
``chunk_size`` taking the same template-parameter slot.

ABI (locked by RFC 0002 item 6 + final_state follow-up):

* op name: ``"ssd_chunk_scan_combined"``
* ``values=[x, dt, A, B, C]`` in exactly that order
* two outputs in this order:
    1. ``Y`` of shape ``[batch, seqlen, n_heads, head_dim]``
    2. ``final_state`` of shape ``[batch, n_heads, head_dim, state_dim]`` —
       the post-last-chunk SSM state, used to seed the step-mode SSM cache.
  Both outputs share the dtype of the inputs.
* compile-time ``chunk_size: Int = 256`` passed as
  ``parameters={"chunk_size": chunk_size}``
* assumes ``ngroups == n_heads`` (caller broadcasts otherwise)
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

from max.graph import Graph, TensorType, TensorValue, ops

logger = logging.getLogger("max.pipelines.mamba2")

_MODULAR_MOJO_MAX_IMPORT_PATH = "MODULAR_MOJO_MAX_IMPORT_PATH"


@functools.cache
def _get_state_space_paths() -> tuple[Path, ...]:
    """Locate the ``state_space.mojoc`` kernel libraries.

    Mirrors ``architectures.mamba.functional_ops._get_state_space_paths``:
    read ``MODULAR_MOJO_MAX_IMPORT_PATH`` (set by Bazel ``mojo_deps`` and
    by the wheel/conda packaging) and pick out the state_space artifact.
    Results are cached since the search is paths-only.
    """
    import_path_env = os.environ.get(_MODULAR_MOJO_MAX_IMPORT_PATH, "")
    if not import_path_env:
        site_packages = Path(__file__).resolve().parents[4]
        wheel_layout = site_packages / "modular"
        conda_layout = site_packages.parent.parent.parent
        for root in (wheel_layout, conda_layout):
            mojo_lib = root / "lib" / "mojo"
            if mojo_lib.is_dir():
                import_path_env = str(mojo_lib)
                break
    if not import_path_env:
        logger.warning(
            "MODULAR_MOJO_MAX_IMPORT_PATH not set for mamba2.functional_ops"
        )
        return ()

    paths: list[Path] = []
    for entry in import_path_env.split(","):
        if not entry.strip():
            continue
        entry_path = Path(entry.strip())
        if not entry_path.is_absolute():
            resolved = Path.cwd() / entry_path
            if not resolved.exists():
                resolved = entry_path
            entry_path = resolved
        if not entry_path.exists():
            continue
        if entry_path.suffix == ".mojoc":
            if "state_space" in entry_path.name:
                paths.append(entry_path.resolve())
            continue
        if entry_path.is_dir():
            for mojoc in entry_path.rglob("*.mojoc"):
                if "state_space" in mojoc.name and (
                    mojoc.is_file() or mojoc.is_symlink()
                ):
                    paths.append(mojoc.resolve())
    logger.info(
        f"mamba2.functional_ops found {len(paths)} state_space paths: {paths}"
    )
    return tuple(paths)


def ssd_chunk_scan_combined(
    x: TensorValue,
    dt: TensorValue,
    A: TensorValue,
    B: TensorValue,
    C: TensorValue,
    chunk_size: int = 256,
    custom_extensions: tuple[Path, ...] | None = None,
) -> tuple[TensorValue, TensorValue]:
    """Mamba2 SSD chunk-scan combined op (prefill, full sequence).

    Builds an ``ops.custom("ssd_chunk_scan_combined", ...)`` node that
    invokes the fused four-stage SSD chunk-scan kernel registered in
    ``max/kernels/src/state_space/ssd_chunk_combined_ops.mojo``.

    Args:
        x: Input projection of shape ``(batch, seqlen, n_heads, head_dim)``.
        dt: Per-token time-step of shape ``(batch, seqlen, n_heads)``.
        A: Diagonal state-decay vector of shape ``(n_heads,)``.
        B: Input projection of shape
            ``(batch, seqlen, n_heads, state_dim)``. The kernel assumes
            ``ngroups == n_heads``; callers with fewer groups must
            broadcast before calling.
        C: Output projection of shape
            ``(batch, seqlen, n_heads, state_dim)``.
        chunk_size: Tokens per chunk for the chunked scan. Must divide
            ``seqlen`` evenly. Forwarded as the compile-time parameter
            ``SSDChunkScanCombined[chunk_size=...]``. Defaults to ``256``
            to match upstream Mamba2 prefill.
        custom_extensions: Optional paths to ``state_space.mojoc``
            kernel libraries. ``None`` triggers auto-discovery from
            ``MODULAR_MOJO_MAX_IMPORT_PATH`` (Bazel/wheel/conda layouts).

    Returns:
        ``(Y, final_state)`` where:

        * ``Y`` is the prefill output of shape
          ``(batch, seqlen, n_heads, head_dim)``.
        * ``final_state`` is the post-last-chunk SSM state of shape
          ``(batch, n_heads, head_dim, state_dim)`` — caller passes this
          into the step-mode SSM cache so decode picks up where prefill
          left off.

        Both share the dtype and device of ``x``.
    """
    if custom_extensions is None:
        custom_extensions = _get_state_space_paths()

    # Register the kernel library with the current Graph so ops.custom can
    # resolve ``ssd_chunk_scan_combined``. ``_import_kernels`` is a no-op
    # if the path is already loaded.
    if custom_extensions:
        Graph.current._import_kernels(list(custom_extensions))

    y_type = TensorType(dtype=x.dtype, shape=x.shape, device=x.device)

    # final_state shape: [batch, n_heads, head_dim, state_dim]. State dim is
    # B's last axis; head_dim is x's last axis; n_heads is x's third axis.
    batch_dim = x.shape[0]
    n_heads_dim = x.shape[2]
    head_dim = x.shape[3]
    state_dim = B.shape[3]
    final_state_type = TensorType(
        dtype=x.dtype,
        shape=[batch_dim, n_heads_dim, head_dim, state_dim],
        device=x.device,
    )

    results = ops.custom(
        "ssd_chunk_scan_combined",
        device=x.device,
        values=[x, dt, A, B, C],
        out_types=[y_type, final_state_type],
        parameters={"chunk_size": int(chunk_size)},
    )
    return results[0].tensor, results[1].tensor
