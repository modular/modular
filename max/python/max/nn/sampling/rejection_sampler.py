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
"""Rejection Sampler custom ops."""

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.kernels import topk_fused_sampling
from max.nn.layer import Module


def _multinomial(
    probs: TensorValue, residual_rand: TensorValue | None = None
) -> TensorValue:
    """Samples from a categorical distribution using the Gumbel-max trick."""
    if residual_rand is not None:
        eps = float(np.finfo(probs.dtype.to_numpy()).eps)
        clamped_uniform = ops.max(
            residual_rand,
            ops.constant(
                eps, dtype=residual_rand.dtype, device=residual_rand.device
            ),
        )
        q = -ops.log(clamped_uniform)
    else:
        eps = float(np.finfo(probs.dtype.to_numpy()).eps)
        uniform_rand_generated = ops.random.uniform(
            like=probs.type,
            range=(eps, 1.0 - eps),
        )
        q = -ops.log(uniform_rand_generated)

    divided = ops.div(probs, q)
    return ops.squeeze(ops.argmax(divided, axis=-1), axis=-1)


def _find_first_rejected(
    rejected: TensorValue, device: DeviceRef
) -> TensorValue:
    """Finds the index of the first True in each row of a boolean mask.

    A sentinel True is appended so that "all accepted" maps to
    ``num_steps`` rather than producing an undefined result.
    """
    with_sentinel = ops.rebind(
        ops.concat(
            [
                rejected,
                ops.broadcast_to(
                    ops.constant(True, dtype=DType.bool, device=device),
                    shape=[Dim("batch_size"), 1],
                ),
            ],
            axis=1,
        ),
        shape=[Dim("batch_size"), Dim("total_num_steps")],
    )
    int_mask = with_sentinel.cast(DType.int32)
    weights = ops.range(
        with_sentinel.shape[1],
        stop=0,
        step=-1,
        out_dim=with_sentinel.shape[1],
        dtype=DType.int64,
        device=device,
    )
    return ops.argmax(int_mask * weights, axis=-1)


class RejectionSampler(Module):
    """Rejection sampler for speculative decoding verification.

    Accepts a draft token when the draft logit for that token does not
    exceed the target logit by more than ``eps``.  Returns
    ``(first_rejected_idx, sampled_target_token)`` - a single recovered
    token at the first rejected position.
    """

    def __init__(
        self,
        device: DeviceRef,
        top_k: int = 1,
        top_p: float = 1,
        temperature: float = 1.0,
        seed: int = 0,
        eps: float = 1e-5,
    ) -> None:
        self.device = device
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.eps = eps
        self.seed = seed

    def __call__(
        self,
        draft_tokens: TensorValue,
        draft_logits_for_sampled_tokens: TensorValue,
        target_logits: TensorValue,
        target_logit_offsets: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        broadcasted_range = ops.broadcast_to(
            ops.range(
                0,
                ops.shape_to_tensor([draft_tokens.shape[1]]).reshape(()),
                1,
                out_dim=Dim("num_steps"),
                device=self.device,
                dtype=DType.int64,
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        logit_offsets = ops.rebind(
            ops.unsqueeze(target_logit_offsets[:-1], axis=-1),
            shape=[Dim("batch_size"), 1],
        )
        sampled_token_offsets = ops.reshape(
            ops.rebind(
                (broadcasted_range + logit_offsets),
                shape=[Dim("batch_size"), Dim("num_steps")],
            ),
            shape=[Dim("batch_size") * Dim("num_steps"), 1],
        )

        target_logits_for_sampled_tokens = ops.reshape(
            ops.gather_nd(
                target_logits,
                ops.concat(
                    [
                        sampled_token_offsets,
                        ops.reshape(
                            draft_tokens,
                            shape=(Dim("batch_size") * Dim("num_steps"), 1),
                        ),
                    ],
                    axis=1,
                ),
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        rejected = ops.rebind(
            draft_logits_for_sampled_tokens
            > target_logits_for_sampled_tokens + self.eps,
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        first_rejected_token = _find_first_rejected(rejected, self.device)

        rejected_offsets = ops.rebind(
            target_logit_offsets[:-1], shape=[Dim("batch_size")]
        ) + ops.squeeze(first_rejected_token, axis=1)

        sampled_target_tokens = topk_fused_sampling(
            logits=ops.gather(target_logits, rejected_offsets, axis=0),
            top_k=self.top_k,
            max_k=self.top_k,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )

        return first_rejected_token, sampled_target_tokens


def _reshape_target_logits(target_logits: TensorValue) -> TensorValue:
    """Reshapes flat target logits to [batch, num_steps+1, vocab]."""
    return ops.reshape(
        ops.rebind(
            target_logits,
            shape=[
                Dim("batch_size") * (Dim("num_steps") + 1),
                Dim("vocab_size"),
            ],
        ),
        shape=[Dim("batch_size"), Dim("num_steps") + 1, Dim("vocab_size")],
    )


def greedy_acceptance_sampler(
    draft_tokens: TensorValue,
    target_logits: TensorValue,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Target-only rejection sampler for speculative decoding.

    - **Greedy** (``greedy=True``): accepts a draft token only when it
    matches the argmax of the target logits.  Recovered tokens are
    the target argmax at every draft position; the bonus token is the
    argmax at the final (+1) position.

    Returns ``(first_rejected_idx, recovered_tokens, bonus_tokens)``
    """
    if draft_tokens.device != target_logits.device:
        raise ValueError(
            "Draft tokens and target logits must be on the same device"
        )
    device = draft_tokens.device

    target_logits_3d = _reshape_target_logits(target_logits)

    all_target_tokens = ops.squeeze(
        ops.argmax(target_logits_3d, axis=-1), axis=-1
    )

    target_tokens_draft = all_target_tokens[:, :-1]
    bonus_tokens = all_target_tokens[:, -1:]

    draft_tokens_rb = ops.rebind(
        draft_tokens, [Dim("batch_size"), Dim("num_steps")]
    )
    rejected = ops.not_equal(
        draft_tokens_rb,
        ops.rebind(target_tokens_draft, [Dim("batch_size"), Dim("num_steps")]),
    )

    first_rejected_idx = ops.squeeze(
        _find_first_rejected(rejected, device), axis=-1
    )

    return first_rejected_idx, target_tokens_draft, bonus_tokens


def stochastic_acceptance_sampler(
    draft_tokens: TensorValue,
    target_logits: TensorValue,
    temperature: TensorValue,
    top_k: TensorValue,
    max_k: TensorValue,
    top_p: TensorValue,
    min_top_p: TensorValue,
    seed: int = 0,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Target-only rejection sampler for speculative decoding.

    - **Stochastic** (``greedy=False``): accepts a draft token with
    probability ``p_target(draft_token)`` after applying temperature
    scaling and softmax.  Recovered tokens are sampled from the
    target distribution; the bonus token is sampled via
    ``topk_fused_sampling``.

    Returns ``(first_rejected_idx, recovered_tokens, bonus_tokens)``
    """
    ops.random.set_seed(seed)

    device = draft_tokens.device

    target_logits_3d = _reshape_target_logits(target_logits)

    draft_verification_logits = target_logits_3d[:, :-1]
    bonus_logits = ops.rebind(
        target_logits_3d[:, -1],
        shape=[Dim("batch_size"), Dim("vocab_size")],
    )

    temp_3d = ops.reshape(temperature, shape=[Dim("batch_size"), 1, 1])
    scaled_logits = draft_verification_logits / temp_3d

    target_probs = ops.softmax(scaled_logits)

    batch_size = draft_tokens.shape[0]
    num_steps = draft_tokens.shape[1]

    batch_indices = ops.broadcast_to(
        ops.reshape(
            ops.range(
                0,
                batch_size,
                1,
                out_dim=Dim("batch_size"),
                device=device,
                dtype=DType.int64,
            ),
            shape=[Dim("batch_size"), 1],
        ),
        shape=[Dim("batch_size"), Dim("num_steps")],
    )

    step_indices = ops.broadcast_to(
        ops.reshape(
            ops.range(
                0,
                num_steps,
                1,
                out_dim=Dim("num_steps"),
                device=device,
                dtype=DType.int64,
            ),
            shape=[1, Dim("num_steps")],
        ),
        shape=[Dim("batch_size"), Dim("num_steps")],
    )

    token_indices = ops.rebind(
        draft_tokens, [Dim("batch_size"), Dim("num_steps")]
    )

    gather_indices = ops.stack(
        [batch_indices, step_indices, token_indices], axis=2
    )

    p_target = ops.gather_nd(target_probs, gather_indices)

    coins = ops.random.uniform(p_target.type)
    rejected = coins >= p_target

    first_rejected_idx = ops.squeeze(
        _find_first_rejected(rejected, device), axis=-1
    )

    recovered_token_ids = _multinomial(target_probs)

    bonus_token_ids = topk_fused_sampling(
        logits=bonus_logits,
        top_k=top_k,
        max_k=max_k,
        temperature=temperature,
        top_p=top_p,
        min_top_p=min_top_p,
    )

    return first_rejected_idx, recovered_token_ids, bonus_token_ids.tensor


class RejectionSamplerWithResiduals(Module):
    """A simple rejection sampler."""

    def __init__(
        self,
        device: DeviceRef,
        top_k: int = 1,
        temperature: float = 1.0,
        eps: float = 1e-10,
        seed: int = 0,
        debug: bool = False,
    ) -> None:
        self.device = device
        self.top_k = top_k
        self.temperature = temperature
        self.eps = eps
        self.debug = debug
        ops.random.set_seed(seed)

    def _get_first_rejected_token_idx(
        self,
        target_logits: TensorValue,
        draft_tokens: TensorValue,
        batch_draft_logits: TensorValue,
        rejection_rand: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        target_logits_reshaped = ops.rebind(
            target_logits,
            shape=[
                Dim("batch_size") * (Dim("num_steps") + 1),
                Dim("vocab_size"),
            ],
        )
        target_logits_without_bonus = ops.reshape(
            target_logits_reshaped,
            shape=[Dim("batch_size"), Dim("num_steps") + 1, Dim("vocab_size")],
        )[:, :-1]

        target_probs = ops.softmax(target_logits_without_bonus)
        draft_probs = ops.softmax(batch_draft_logits)

        batch_size = batch_draft_logits.shape[0]
        num_steps = batch_draft_logits.shape[1]

        batch_indices = ops.broadcast_to(
            ops.reshape(
                ops.range(
                    0,
                    batch_size,
                    1,
                    out_dim=Dim("batch_size"),
                    device=self.device,
                    dtype=DType.int64,
                ),
                shape=[Dim("batch_size"), 1],
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        step_indices = ops.broadcast_to(
            ops.reshape(
                ops.range(
                    0,
                    num_steps,
                    1,
                    out_dim=Dim("num_steps"),
                    device=self.device,
                    dtype=DType.int64,
                ),
                shape=[1, Dim("num_steps")],
            ),
            shape=[Dim("batch_size"), Dim("num_steps")],
        )

        token_indices = ops.rebind(
            draft_tokens, [Dim("batch_size"), Dim("num_steps")]
        )

        gather_indices = ops.stack(
            [batch_indices, step_indices, token_indices], axis=2
        )

        target_probs_for_sampled_tokens = ops.gather_nd(
            target_probs, gather_indices
        )
        draft_probs_for_sampled_tokens = ops.gather_nd(
            draft_probs, gather_indices
        )
        ratio = target_probs_for_sampled_tokens / (
            draft_probs_for_sampled_tokens + self.eps
        )

        if rejection_rand:
            uniform_rand_values = rejection_rand
        else:
            uniform_rand_values = ops.random.uniform(ratio.type)

        capped_ratio = ops.min(
            ratio, ops.constant(1, dtype=DType.float32, device=self.device)
        )

        rejected = uniform_rand_values >= capped_ratio
        rejected_with_sentinel = ops.concat(
            [
                rejected,
                ops.broadcast_to(
                    ops.constant(True, dtype=DType.bool, device=self.device),
                    shape=[Dim("batch_size"), 1],
                ),
            ],
            axis=1,
        )

        rejected_with_sentinel = rejected_with_sentinel.cast(DType.int32)
        # argmax is not reliable for getting the first max occurrence when dealing with int tensors with [0,1] values, so we weight them here to get the first occurrence.
        # TODO: remove this when/if KERN-1862 is resolved
        argmax_weights = ops.range(
            rejected_with_sentinel.shape[1],
            stop=0,
            step=-1,
            out_dim=rejected_with_sentinel.shape[1],
            dtype=DType.int64,
            device=self.device,
        )
        first_rejected_index = ops.argmax(
            rejected_with_sentinel * argmax_weights, axis=-1
        )

        return (
            ops.squeeze(first_rejected_index, axis=-1),
            draft_probs,
            target_probs,
        )

    def _get_recovered_probs(
        self,
        target_probs: TensorValue,
        draft_probs: TensorValue,
    ) -> TensorValue:
        difference = target_probs - draft_probs
        float_tiny = float(np.finfo(difference.dtype.to_numpy()).tiny)
        f = ops.max(
            difference,
            ops.constant(
                float_tiny, dtype=difference.dtype, device=self.device
            ),
        )

        recovered_probs = f / ops.reshape(
            ops.sum(f), shape=[-1, Dim("num_steps"), 1]
        )

        return recovered_probs

    def __call__(
        self,
        draft_tokens: TensorValue,
        draft_logits_for_sampled_tokens: TensorValue,
        target_logits: TensorValue,
        target_logit_offsets: TensorValue,
        all_draft_logits: TensorValue,
        rejection_rand: TensorValue | None = None,
        residual_rand: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        batch_draft_logits = ops.permute(
            all_draft_logits,
            [1, 0, 2],
        )
        first_rejected_token_idx, draft_probs, target_probs = (
            self._get_first_rejected_token_idx(
                target_logits,
                draft_tokens,
                batch_draft_logits,
                rejection_rand,
            )
        )
        recovered_probs = self._get_recovered_probs(target_probs, draft_probs)

        if residual_rand:
            recovered_token_ids = _multinomial(recovered_probs, residual_rand)
        else:
            recovered_token_ids = _multinomial(recovered_probs)

        bonus_indices = ops.rebind(
            target_logit_offsets[1:] - 1, shape=[Dim("batch_size")]
        )
        bonus_logits = ops.gather(target_logits, bonus_indices, axis=0)
        bonus_token_ids = topk_fused_sampling(
            logits=bonus_logits,
            top_k=self.top_k,
            max_k=self.top_k,
            temperature=self.temperature,
        )
        return (
            first_rejected_token_idx,
            recovered_token_ids,
            bonus_token_ids.tensor,
        )
