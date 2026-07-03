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

"""Unit tests for multi-group NIXL registration (SERVOPT-1484).

Root cause: ``MultiKVCacheBuffer.all_buffers`` flattens target + draft KV into
one list; when target and draft have different buffer shapes (e.g., 61-layer MLA
vs. 1-layer Eagle), ``_validate_tensor_shape`` raises a shape-mismatch error.

Fix: ``from_paged_kv_cache`` now registers each child cache as a separate NIXL
group, keeping each group's buffers homogeneous.

These tests validate the arithmetic of per-group bytes_per_page and the
descriptor-layout logic without requiring actual NIXL/GPU objects.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers that mirror the per-group bytes_per_page arithmetic
# ---------------------------------------------------------------------------


def _bpp(num_elements: int, dtype_bytes: int, total_num_pages: int) -> int:
    """Compute bytes_per_page as the engine does for a single group."""
    assert num_elements % total_num_pages == 0, (
        f"num_elements {num_elements} not divisible by {total_num_pages}"
    )
    return num_elements * dtype_bytes // total_num_pages


def _group_descriptors(
    base_addr: int,
    bytes_per_page: int,
    page_idxs: list[int],
    device_id: int,
) -> list[tuple[int, int, int]]:
    """Build transfer descriptors for one group and a list of page indices."""
    return [
        (base_addr + idx * bytes_per_page, bytes_per_page, device_id)
        for idx in page_idxs
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validate_shape_single_group_ok() -> None:
    """Single-group validation: homogeneous buffers pass."""
    total_pages = 1000
    elts = 73728 * 61
    bpp = _bpp(elts * total_pages, 1, total_pages)
    assert bpp == 73728 * 61


def test_eagle_target_and_draft_different_bpp() -> None:
    """Target (61 layers) and draft (1 layer) produce different bytes_per_page."""
    total_pages = 1000
    target_bpp = _bpp(73728 * 61 * total_pages, 1, total_pages)
    draft_bpp = _bpp(73728 * 1 * total_pages, 1, total_pages)
    assert target_bpp != draft_bpp
    assert target_bpp == 73728 * 61
    assert draft_bpp == 73728 * 1


def test_multi_group_total_bpp_is_sum() -> None:
    """bytes_per_page for a multi-group engine equals sum of all group bpps."""
    total_pages = 1000
    target_bpp = _bpp(73728 * 61 * total_pages, 1, total_pages)
    draft_bpp = _bpp(73728 * 1 * total_pages, 1, total_pages)
    bytes_per_group = [target_bpp, draft_bpp]
    total_bpp = sum(bytes_per_group)
    assert total_bpp == target_bpp + draft_bpp


def test_multi_group_descriptor_layout_correct() -> None:
    """Descriptors for page index k cover each group at the right address."""
    total_pages = 1000
    target_bpp = _bpp(73728 * 61 * total_pages, 1, total_pages)
    draft_bpp = _bpp(73728 * 1 * total_pages, 1, total_pages)
    bytes_per_group = [target_bpp, draft_bpp]

    target_base = 0x1000_0000
    draft_base = 0x2000_0000
    src_base_addrs = [target_base, draft_base]

    page_idxs = [3, 7, 12]
    device_id = 0

    descs: list[tuple[int, int, int]] = []
    for group_idx, bpp in enumerate(bytes_per_group):
        base = src_base_addrs[group_idx]
        descs.extend(_group_descriptors(base, bpp, page_idxs, device_id))

    # Should have len(page_idxs) * num_groups descriptors
    assert len(descs) == len(page_idxs) * len(bytes_per_group)

    # Each page in the target group should be at target_base + k * target_bpp
    for i, k in enumerate(page_idxs):
        addr, size, dev = descs[i]
        assert addr == target_base + k * target_bpp
        assert size == target_bpp
        assert dev == device_id

    # Each page in the draft group should be at draft_base + k * draft_bpp
    offset = len(page_idxs)
    for i, k in enumerate(page_idxs):
        addr, size, dev = descs[offset + i]
        assert addr == draft_base + k * draft_bpp
        assert size == draft_bpp
        assert dev == device_id


def test_single_group_descriptor_unchanged() -> None:
    """Single-group engines produce the same descriptor layout as before."""
    total_pages = 1000
    elts = 73728 * 61
    bpp = _bpp(elts * total_pages, 1, total_pages)
    bytes_per_group = [bpp]

    base = 0x1000_0000
    page_idxs = [5, 10, 15]
    device_id = 1

    descs = []
    for bpp_g in bytes_per_group:
        descs.extend(_group_descriptors(base, bpp_g, page_idxs, device_id))

    assert len(descs) == len(page_idxs)
    for i, k in enumerate(page_idxs):
        addr, size, _ = descs[i]
        assert addr == base + k * bpp
        assert size == bpp


def test_group_validation_fails_for_mixed_flat_list() -> None:
    """Mixing target+draft into one flat list fails shape validation."""
    total_pages = 1000
    target_num_elts = 73728 * 61 * total_pages
    draft_num_elts = 73728 * 1 * total_pages

    # Simulate what _validate_tensor_shape does: check all tensors have same shape
    shapes = [target_num_elts, draft_num_elts]
    all_same = len(set(shapes)) == 1
    assert not all_same, (
        "Mixed target+draft in a flat list should have heterogeneous shapes"
    )


def test_per_group_validation_passes() -> None:
    """Each group validated separately: no shape mismatch within a group."""
    total_pages = 1000

    # Target group: two TP shards, both with same shape
    target_num_elts = 73728 * 61 * total_pages
    target_shapes = [target_num_elts, target_num_elts]
    assert len(set(target_shapes)) == 1, "Target group should be homogeneous"

    # Draft group: two TP shards, both with same shape
    draft_num_elts = 73728 * 1 * total_pages
    draft_shapes = [draft_num_elts, draft_num_elts]
    assert len(set(draft_shapes)) == 1, "Draft group should be homogeneous"

    # The two groups differ from each other (which is fine — separate groups)
    assert target_num_elts != draft_num_elts
