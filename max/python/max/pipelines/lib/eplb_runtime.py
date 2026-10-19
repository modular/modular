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
"""The EPLB state a pipeline model owns, behind one object.

A model adds EPLB with four calls, in load order::

    eplb = EplbRuntime(pipeline_config.runtime, metadata, devices)
    eplb.install_placement(ep_config, n_groups=...)  # before EP init
    state_dict = eplb.remap_weights(nn_model, state_dict)
    eplb.bind(ep_comm_initializer)  # after EP init

and reads ``eplb.accumulator`` for routing profiling.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import TypeVar

from max.driver import Buffer, Device, is_virtual_device_mode
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.layer import Module
from max.nn.moe import apply_eplb_placement, router_fold_supported

from .eplb_stats import (
    EplbPlacement,
    EplbStatsAccumulator,
    EplbStatsMetadata,
    EplbStatsSnapshot,
    eplb_planner_for,
)
from .pipeline_runtime_config import PipelineRuntimeConfig

logger = logging.getLogger("max.pipelines")

_T = TypeVar("_T")

EPLB_STATS_ENV = "MAX_SERVE_EPLB_STATS"
"""Path to a routing snapshot; set by ``--eplb-stats``."""


class EplbRuntime:
    """Owns one model's EPLB placement, its device tables, and profiling."""

    def __init__(
        self,
        runtime: PipelineRuntimeConfig,
        metadata: EplbStatsMetadata,
        devices: Sequence[Device],
    ) -> None:
        """Initializes the runtime.

        Args:
            runtime: The pipeline runtime config carrying the EPLB options.
            metadata: The routing-histogram shape of the model.
            devices: The model's local devices.
        """
        self._runtime = runtime
        self._devices = list(devices)
        self.metadata = metadata
        self.placement: EplbPlacement | None = None
        self._fold_router = False
        # Virtual exports stage devices that cannot service histogram
        # buffers. The graph's histogram inputs are still exported, so a
        # boot with the same env compiles the same graph; the serving
        # process constructs its own accumulator from real devices.
        self.accumulator: EplbStatsAccumulator | None = (
            EplbStatsAccumulator(metadata=metadata, devices=self._devices)
            if runtime.eplb_profile and not is_virtual_device_mode()
            else None
        )

    def install_placement(
        self, ep_config: EPConfig, *, n_groups: int
    ) -> EplbPlacement | None:
        """Plans a placement from the configured snapshot, if any.

        Sizes ``ep_config`` for the placement's physical slots, so call this
        before the model and the EP communication buffers are built.

        Args:
            ep_config: The EP config to update in place.
            n_groups: Number of expert groups the router selects from.

        Returns:
            The placement, or ``None`` when no snapshot is configured.

        Raises:
            ValueError: If the snapshot does not match the model.
        """
        path = os.environ.get(EPLB_STATS_ENV)
        if not path:
            return None
        with open(path) as f:
            snap = EplbStatsSnapshot.from_dict(json.load(f))
        md = self.metadata
        if (
            snap.metadata.num_layers != md.num_layers
            or snap.metadata.num_logical_experts != md.num_logical_experts
        ):
            raise ValueError(
                f"EPLB snapshot {snap.metadata} doesn't match model {md}"
            )
        logger.info(
            "EPLB: loaded snapshot from %s "
            "(num_moe_layers=%d, num_logical_experts=%d, total_tokens=%d, "
            "planner=%s)",
            path,
            snap.metadata.num_moe_layers,
            snap.metadata.num_logical_experts,
            snap.total_tokens,
            self._runtime.eplb_planner,
        )

        placement = EplbPlacement.from_snapshot(
            snap,
            ep_size=ep_config.n_gpus_per_node * ep_config.n_nodes,
            n_nodes=ep_config.n_nodes,
            n_groups=n_groups,
            eplb_redundant_experts_per_gpu=self._runtime.eplb_redundant_experts_per_gpu,
            planner=eplb_planner_for(self._runtime.eplb_planner),
        )
        ep_config.eplb_enabled = True
        ep_config.num_moe_layers = placement.phy2log.shape[0]
        ep_config.max_replicas = placement.max_replicas
        ep_config.num_logical_experts = placement.log2phy.shape[1]
        ep_config.n_experts = placement.num_phy
        ep_config.eplb_phy2log_plan = placement.phy2log
        # Decided here, before the model and its EP manager copy the config,
        # so every copy agrees with how remap_weights lays out the routers.
        self._fold_router = (
            self._runtime.eplb_fold_router
            and not self._runtime.eplb_profile
            and int(placement.logcnt.max()) == 1
            and n_groups == 1
        )
        ep_config.eplb_router_folded = self._fold_router
        logger.info(
            "EPLB: placement installed (num_phy=%d, max_replicas=%d, "
            "num_moe_layers=%d)",
            placement.num_phy,
            placement.max_replicas,
            placement.phy2log.shape[0],
        )
        self.placement = placement
        return placement

    def remap_weights(
        self, model: Module, state_dict: Mapping[str, _T]
    ) -> Mapping[str, _T | Buffer | WeightData]:
        """Points every physical expert slot at its logical expert's weights.

        For a permutation-only placement, also reorders each router's
        expert rows (see ``eplb_fold_router``) to match what
        :meth:`install_placement` told the EP config. Returns ``state_dict``
        unchanged when no placement is installed.

        Raises:
            ValueError: If the placement was to be folded but some router
                keeps an expert-indexed weight a row reorder cannot cover.
        """
        if self.placement is None:
            return state_dict
        phy2log = self.placement.phy2log
        if self._fold_router and not router_fold_supported(model, phy2log):
            raise ValueError(
                "EPLB cannot fold this placement into the model's routers; "
                "set eplb_fold_router=False"
            )
        logger.info(
            "EPLB: %s",
            "placement folded into the routers; no id remap at dispatch"
            if self._fold_router
            else "routers emit logical ids; dispatch remaps them",
        )
        return apply_eplb_placement(
            model, state_dict, phy2log, fold_router=self._fold_router
        )

    def bind(self, ep_comm_initializer: EPCommInitializer | None) -> None:
        """Hands the placement's device tables to the EP model inputs.

        A no-op without a placement, or without an initializer (virtual
        device compilation never executes the graph).
        """
        if self.placement is None or ep_comm_initializer is None:
            return
        ep_comm_initializer.set_eplb_placement(
            [
                Buffer.from_numpy(self.placement.log2phy).to(d)
                for d in self._devices
            ],
            [
                Buffer.from_numpy(self.placement.logcnt).to(d)
                for d in self._devices
            ],
        )
