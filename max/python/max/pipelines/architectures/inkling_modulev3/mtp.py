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
"""Chained Inkling MTP depths on ModuleV3: one full decoder block per step.

The depths embed and project through the target's embedding and head, so they
hold neither.
"""

from __future__ import annotations

from collections.abc import Mapping

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferType, BufferValue, DeviceRef, ops
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.linear import Linear
from max.nn.sequential import ModuleList
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, NamedMapping, Partial
from max.tensor.sharding.action import PerShard

from ..inkling.model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    InklingConfig,
)
from ..inkling.state_cache import (
    CONV_STATE_DTYPE,
    ConvSite,
    InklingConvStateLayout,
)
from .inkling import InklingDecoderLayer, _ConvSites, _log_scaling_tau
from .layers.placement import as_partial, as_replicated, reduce_partial
from .layers.short_convolution import InklingRMSNorm

__all__ = ["InklingMTPDepth", "InklingMultiTokenPredictor"]


class InklingMTPDepth(Module[..., Tensor]):
    """One MTP depth: dual RMSNorm, a 2H to H fuse, then a dense Inkling block."""

    def __init__(
        self,
        config: InklingConfig,
        *,
        depth_idx: int,
        is_local: bool,
        kv_params: MHAKVCacheParams,
        cache_layer_idx: int,
    ) -> None:
        """Builds the depth's norms, fuse and decoder block.

        The block never commits its conv inputs.
        """
        text_config = config.text_config
        hidden = text_config.hidden_size
        self.kv_key = LOCAL_ATTENTION if is_local else GLOBAL_ATTENTION
        self.cache_layer_idx = cache_layer_idx
        self.hidden_norm = InklingRMSNorm(hidden, eps=text_config.rms_norm_eps)
        self.embed_norm = InklingRMSNorm(hidden, eps=text_config.rms_norm_eps)
        self.input_proj = Linear(2 * hidden, hidden, bias=False)
        self.decoder_layer = InklingDecoderLayer(
            text_config=text_config,
            layer_idx=depth_idx,
            kv_params=kv_params,
            num_devices=len(config.devices),
            is_local=is_local,
            force_dense_mlp=True,
            commit_conv_state=False,
        )

    def forward(
        self,
        embeds: Tensor,
        hidden: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        log_scaling: Tensor,
        conv: _ConvSites,
        hidden_states_first: bool,
    ) -> Tensor:
        """Fuses ``hidden`` with the token embeddings and runs the block."""
        h = self.hidden_norm(hidden)
        e = F.rebind(self.embed_norm(embeds), h.shape)
        x = self.input_proj(
            F.concat([h, e] if hidden_states_first else [e, h], axis=-1)
        )
        x, delta = self.decoder_layer(
            x,
            as_partial(F.zeros_like(x), x.mesh),
            kv_collection,
            input_row_offsets,
            positions,
            log_scaling,
            conv,
            F.constant(self.cache_layer_idx, DType.uint32, device=CPU()),
        )
        return x + reduce_partial(delta)


class InklingMultiTokenPredictor(Module[..., Tensor]):
    """Stacked MTP depths sharing the target's embedding table and head."""

    def __init__(
        self,
        config: InklingConfig,
        n_depths: int,
        kv_params: MultiKVCacheParams,
    ) -> None:
        """Builds ``n_depths`` depths, each on its flavor's cache leaf."""
        assert config.mtp is not None
        text_config = config.text_config
        self.text_config = text_config
        self.kv_params = kv_params
        self.hidden_states_first = config.mtp.hidden_states_first
        is_local = config.mtp.local_flags(n_depths)
        self.conv_layout = InklingConvStateLayout.from_local_flags(
            text_config, is_local, tp_size=len(config.devices)
        )
        layers = []
        counts = {GLOBAL_ATTENTION: 0, LOCAL_ATTENTION: 0}
        for depth_idx, local in enumerate(is_local):
            key = LOCAL_ATTENTION if local else GLOBAL_ATTENTION
            layers.append(
                InklingMTPDepth(
                    config,
                    depth_idx=depth_idx,
                    is_local=local,
                    kv_params=kv_params.child(key, MHAKVCacheParams),
                    cache_layer_idx=counts[key],
                )
            )
            counts[key] += 1
        self.layers = ModuleList(layers)
        self.chain_norm = (
            InklingRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
            if config.mtp.chain_hidden_post_norm
            else None
        )

    def forward_depth(
        self,
        depth_idx: int,
        embeds: Tensor,
        hidden: Tensor,
        kv_collections: Mapping[str, PagedCacheValues],
        input_row_offsets: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Runs depth ``depth_idx`` over one ragged batch, then the chain norm."""
        depth = self.layers[depth_idx]
        assert isinstance(depth, InklingMTPDepth)
        log_scaling = as_replicated(
            _log_scaling_tau(
                positions,
                alpha=self.text_config.log_scaling_alpha,
                n_floor=self.text_config.log_scaling_n_floor,
            ),
            hidden.mesh,
        )
        h = depth(
            embeds,
            hidden,
            kv_collections[depth.kv_key],
            input_row_offsets,
            positions,
            log_scaling,
            self._zero_slot(depth_idx, input_row_offsets),
            self.hidden_states_first,
        )
        if self.chain_norm is not None:
            h = self.chain_norm(h)
        return h

    def forward(self, *args: object) -> Tensor:
        """Raises, because each depth runs through :meth:`forward_depth`."""
        raise TypeError(
            "InklingMultiTokenPredictor runs one depth at a time, through"
            " forward_depth"
        )

    def _zero_slot(
        self, depth_idx: int, input_row_offsets: Tensor
    ) -> _ConvSites:
        """Returns one depth's conv pools and slot tables: one zeroed slot.

        The depths never commit their conv inputs, so their taps before a
        chunk read zero, which a pool the graph creates zeroed holds. The
        pools are placed as the backbone's are.
        """
        mesh = input_row_offsets.mesh
        replicated = DeviceMapping.replicated(mesh)
        devices = PerShard(
            tuple(DeviceRef.from_device(device) for device in mesh.devices)
        )
        pools = []
        for site, channels in zip(
            ConvSite, self.conv_layout.layers[depth_idx], strict=True
        ):
            mapping = (
                NamedMapping(mesh, (None, None, TP))
                if site in (ConvSite.K, ConvSite.V)
                else DeviceMapping(mesh, (Partial(),))
            )
            pools.append(
                F.call_on_mesh(
                    _zero_pool,
                    mesh,
                    out_specs=replicated if mesh.num_devices == 1 else mapping,
                )(devices, [1, self.conv_layout.ring_len, channels])
            )
        rows = F.call_on_mesh(_zero_rows, mesh, out_specs=replicated)(
            input_row_offsets
        )
        zero_row = F.constant(0, DType.uint32, device=CPU())
        return _ConvSites(
            pools=pools,
            rows=[rows] * len(ConvSite),
            layer_rows=[zero_row] * len(ConvSite),
        )


def _zero_pool(device: DeviceRef, shape: list[int]) -> BufferValue:
    """Creates one device's conv pool, zeroed."""
    return ops.buffer_create(
        BufferType(CONV_STATE_DTYPE, shape, device=device), init_value=0.0
    )


def _zero_rows(offsets: Tensor) -> Tensor:
    """Points each request of one device's batch at the pool's one slot."""
    return F.broadcast_to(
        F.constant(0, DType.uint32, device=offsets.device),
        [1, offsets.shape[0] - 1],
    )
