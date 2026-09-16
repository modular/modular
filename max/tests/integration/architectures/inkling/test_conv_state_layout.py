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
"""The pool leaves an Inkling conv state occupies."""

from __future__ import annotations

from max.pipelines.architectures.inkling.state_cache import (
    CONV_STATE_DTYPE,
    ConvSite,
    InklingConvStateLayout,
)

_STATE_LEN = 3
_LOCAL_KV = 256
_GLOBAL_KV = 1024
_RESIDUAL = 4096

# Layers 1 and 3 are sliding-window; 0 and 2 are global.
_IS_LOCAL = (False, True, False, True)


def _layout() -> InklingConvStateLayout:
    return InklingConvStateLayout(
        state_len=_STATE_LEN,
        layers=tuple(
            (kv, kv, _RESIDUAL, _RESIDUAL)
            for kv in (
                _LOCAL_KV if local else _GLOBAL_KV for local in _IS_LOCAL
            )
        ),
        is_local=_IS_LOCAL,
    )


def test_a_leaf_per_site_and_layer_kind() -> None:
    """Only K and V vary, and only between the kinds, so eight leaves."""
    regions = {r.leaf_id: r for r in _layout().regions()}

    assert set(regions) == {
        f"conv/{kind}/{site.name.lower()}"
        for kind in ("global", "local")
        for site in ConvSite
    }
    assert all(r.num_layers == 2 for r in regions.values())
    assert all(r.dtype == CONV_STATE_DTYPE for r in regions.values())


def test_each_leaf_is_uniformly_shaped() -> None:
    """The whole point: a leaf the pool can tile with one row shape."""
    regions = {r.leaf_id: r for r in _layout().regions()}

    assert regions["conv/global/k"].row_shape == (_GLOBAL_KV, _STATE_LEN)
    assert regions["conv/local/k"].row_shape == (_LOCAL_KV, _STATE_LEN)
    assert regions["conv/global/attn_out"].row_shape == (_RESIDUAL, _STATE_LEN)
    assert regions["conv/local/mlp_out"].row_shape == (_RESIDUAL, _STATE_LEN)


def test_a_layer_addresses_its_own_kind_by_ordinal() -> None:
    """Row index within a leaf is the layer's ordinal among its own kind.

    ``row_for`` names the leaf by position, so this also pins the contract
    the graph reads it under: that position indexes the same leaf in
    :meth:`regions` as it does in a device's ``leaves`` tuple.
    """
    layout = _layout()
    regions = layout.regions()

    def addressed(layer_idx: int, site: ConvSite) -> tuple[str, int]:
        leaf, row = layout.row_for(layer_idx, site)
        return regions[leaf].leaf_id, row

    assert addressed(0, ConvSite.K) == ("conv/global/k", 0)
    assert addressed(2, ConvSite.K) == ("conv/global/k", 1)
    assert addressed(1, ConvSite.V) == ("conv/local/v", 0)
    assert addressed(3, ConvSite.V) == ("conv/local/v", 1)


def test_every_site_addresses_a_leaf_that_holds_its_row() -> None:
    """No layer/site pair addresses past its leaf, or into another's."""
    layout = _layout()
    regions = layout.regions()

    for layer_idx in range(layout.num_layers):
        kind = "local" if layout.is_local[layer_idx] else "global"
        for site in ConvSite:
            leaf, row = layout.row_for(layer_idx, site)
            region = regions[leaf]
            assert region.leaf_id == f"conv/{kind}/{site.name.lower()}"
            assert 0 <= row < region.num_layers


def test_the_leaves_hold_exactly_what_the_pools_did() -> None:
    """Regrouping moves the bytes around; it must not change the total."""
    layout = _layout()

    assert sum(r.bytes_per_page for r in layout.regions()) == (
        layout.bytes_per_request()
    )


def test_a_uniform_model_still_splits_by_kind() -> None:
    """All-global is one kind, so four leaves rather than eight."""
    layout = InklingConvStateLayout(
        state_len=_STATE_LEN,
        layers=((_GLOBAL_KV, _GLOBAL_KV, _RESIDUAL, _RESIDUAL),) * 3,
        is_local=(False, False, False),
    )

    regions = layout.regions()

    assert {r.leaf_id for r in regions} == {
        f"conv/global/{site.name.lower()}" for site in ConvSite
    }
    assert all(r.num_layers == 3 for r in regions)
