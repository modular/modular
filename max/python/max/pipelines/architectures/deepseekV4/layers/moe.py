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

"""DeepSeek-V4 MoE, with hash routing on the first ``num_hash_layers`` layers.

Reference: ``inference/model.py`` classes ``Gate``, ``Expert`` and ``MoE``.
Every V4 layer is MoE -- there is no dense prefix -- and every layer has one
shared expert on top of the ``num_experts_per_tok`` routed ones.

Routing has two shapes, and which one a layer uses is not a property of the
gate weight but of the layer index:

* Layers below ``num_hash_layers`` (0, 1 and 2) read their expert indices
  straight out of ``gate.tid2eid[token_id]``. The table is different on each of
  the three layers. They still compute gate scores and still have a
  ``gate.weight``, because the scores supply the routing *weights* -- only the
  choice of expert comes from the table.
* The rest score normally and take the top-k of the bias-shifted scores.

The two are mutually exclusive in the checkpoint: hash layers have no
``gate.bias``, scored layers have no ``gate.tid2eid``.

Three details from ``Gate.forward`` that are each a plausible place to be
wrong:

* The bias shifts the scores *for selection only*. The weights are gathered
  from the unshifted scores.
* The normalization divides by the sum of the gathered weights, which happens
  after the gather, not by the sum over all experts.
* ``routed_scaling_factor`` multiplies after the normalization, so it does not
  cancel.

``config.json`` says ``topk_method: noaux_tc``, which is vestigial: the
reference takes a plain top-k with no group limiting, and ``n_group`` /
``topk_group`` are not in the config at all.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear

from ..model_config import DeepseekV4Config


def sqrt_softplus(x: TensorValue) -> TensorValue:
    """``F.softplus(x).sqrt()`` -- ``scoring_func: sqrtsoftplus``.

    MAX has no ``softplus``, so it is written in the numerically stable form:
    ``max(x, 0) + log1p(exp(-|x|))``. The naive ``log(1 + exp(x))`` overflows
    for large positive scores, which is exactly where a routed expert's score
    lives.
    """
    zero = ops.constant(0.0, DType.float32, x.device)
    softplus = ops.max(x, zero) + ops.log1p(ops.exp(-ops.abs(x)))
    return ops.sqrt(softplus)


class DeepseekV4Expert(Module):
    """SwiGLU FFN. ``w1`` gate, ``w3`` up, ``w2`` down.

    The routing weight is applied *inside*, between the activation and ``w2``.
    Scaling the output instead would be the same algebra but not the same
    arithmetic: the reference multiplies in float32 and casts to the model
    dtype afterwards, so the rounding lands on the scaled value.
    """

    def __init__(self, config: DeepseekV4Config, device: DeviceRef) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.w1 = Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            config.dtype,
            device,
        )
        self.w2 = Linear(
            config.moe_intermediate_size,
            config.hidden_size,
            config.dtype,
            device,
        )
        self.w3 = Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            config.dtype,
            device,
        )

    def __call__(
        self, x: TensorValue, weights: TensorValue | None = None
    ) -> TensorValue:
        gate = ops.cast(self.w1(x), DType.float32)
        up = ops.cast(self.w3(x), DType.float32)
        if self.swiglu_limit > 0:
            limit = ops.constant(self.swiglu_limit, DType.float32, gate.device)
            # Asymmetric on purpose: the reference clamps ``up`` on both sides
            # but ``gate`` only from above.
            up = ops.min(ops.max(up, -limit), limit)
            gate = ops.min(gate, limit)
        h = ops.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(ops.cast(h, x.dtype))


class DeepseekV4Gate(Module):
    """Expert selection and routing weights for one layer."""

    def __init__(
        self, config: DeepseekV4Config, layer_idx: int, device: DeviceRef
    ) -> None:
        super().__init__()
        self.is_hash_routed = config.layer_is_hash_routed(layer_idx)
        self.topk = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.route_scale = config.routed_scaling_factor

        self.weight = Weight(
            name="weight",
            dtype=config.dtype,
            shape=(config.n_routed_experts, config.hidden_size),
            device=device,
        )
        if self.is_hash_routed:
            # int64 in the checkpoint. ``inference/model.py:564`` declares
            # int32, but that is the declaration; the stored tensor is I64 in
            # both the full and the minimized model, and reading it as int32
            # gives misaligned indices.
            self.tid2eid = Weight(
                name="tid2eid",
                dtype=DType.int64,
                shape=(config.vocab_size, config.num_experts_per_tok),
                device=device,
            )
        else:
            self.bias = Weight(
                name="bias",
                dtype=DType.float32,
                shape=(config.n_routed_experts,),
                device=device,
            )

    def __call__(
        self, x: TensorValue, token_ids: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """``[b, s, d]`` and ``[b, s]`` -> weights and indices, both ``[b, s, k]``."""
        scores = sqrt_softplus(
            ops.matmul(
                ops.cast(x, DType.float32),
                ops.transpose(ops.cast(self.weight, DType.float32), 0, 1),
            )
        )
        if self.is_hash_routed:
            indices = ops.gather(self.tid2eid, token_ids, axis=0)
        else:
            _, indices = ops.top_k(scores + self.bias, self.topk, axis=-1)
        indices = ops.cast(indices, DType.int32)

        # Per-token gather along the expert axis, so batch_dims covers both
        # batch and sequence.
        weights = ops.gather_nd(
            scores, ops.unsqueeze(indices, -1), batch_dims=2
        )
        weights = weights / ops.sum(weights, axis=-1)
        return weights * self.route_scale, indices


class DeepseekV4MoE(Module):
    """``num_experts_per_tok`` routed experts plus one shared expert."""

    def __init__(
        self, config: DeepseekV4Config, layer_idx: int, device: DeviceRef
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_routed_experts = config.n_routed_experts
        self.gate = DeepseekV4Gate(config, layer_idx, device)
        self.experts = LayerList(
            [
                DeepseekV4Expert(config, device)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = DeepseekV4Expert(config, device)

    def __call__(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """``[b, s, hidden]`` in, same out.

        Dense dispatch: every expert sees every token, weighted by zero where
        it was not selected. Per MXSERV-502 kernel performance is out of scope,
        and a gather-based dispatch needs a dynamic shape per expert, which the
        graph cannot express. On the 8-expert minimized model this costs 8/6 of
        the routed work; on the 256-expert model it would cost 42x, so this is
        the piece to replace first if the full checkpoint is ever run.

        The scatter from ``[b, s, k]`` slots to ``[b, s, n_experts]`` sums when
        two slots name the same expert. The reference's ``y[idx] += v`` would
        instead keep only the last of them, but no such row exists: every
        ``tid2eid`` row in the checkpoint holds ``k`` distinct experts.
        """
        weights, indices = self.gate(x, token_ids)
        experts = ops.range(
            0,
            self.n_routed_experts,
            1,
            out_dim=self.n_routed_experts,
            device=indices.device,
            dtype=DType.int32,
        )
        selected = ops.cast(
            ops.unsqueeze(indices, -1) == experts, DType.float32
        )
        per_expert = ops.squeeze(
            ops.sum(ops.unsqueeze(weights, -1) * selected, axis=2), axis=2
        )

        y = ops.cast(self.shared_experts(x), DType.float32)
        for i, expert in enumerate(self.experts):
            y = y + ops.cast(
                expert(x, per_expert[..., i : i + 1]), DType.float32
            )
        return ops.cast(y, x.dtype)
