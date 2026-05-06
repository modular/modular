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
"""Tests for rejection_sampler_with_residuals (speculative decoding)."""

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.lib import rejection_sampler_with_residuals


def rejection_sampler_reference(  # noqa: ANN201
    target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    draft_token_ids: torch.Tensor,  # [batch_size, k]
    rejection_rand: torch.Tensor,  # [batch_size, k]
    residual_rand: torch.Tensor | None = None,  # [batch_size * k, vocab_size]
):
    """Rejection sampler reference implementation."""

    def _get_first_rejected_token_idx(
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size, device=target_probs.device)[
            :, None
        ]
        probs_indices = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[
            batch_indices, probs_indices, draft_token_ids
        ]

        # shape [batch_size, k]
        selected_target_probs = target_probs[
            batch_indices, probs_indices, draft_token_ids
        ]

        ratio = selected_target_probs / selected_draft_probs
        capped_ratio = torch.minimum(
            ratio,
            torch.full((1,), 1, device=target_probs.device),
        )
        rejected = (rejection_rand >= capped_ratio).long()
        rejected_with_sentinel = torch.concat(
            [rejected, torch.ones((batch_size, 1))],
            dim=-1,
        )

        first_rejected_token_idx = torch.argmax(rejected_with_sentinel, dim=-1)
        return first_rejected_token_idx

    def _get_recovered_probs(
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
    ) -> torch.Tensor:
        _, k, _ = draft_probs.shape

        # shape [batch_size, k, vocab_size]
        difference = target_probs - draft_probs

        # shape [batch_size, k, vocab_size]
        f = torch.clamp(difference, min=torch.finfo(difference.dtype).tiny)

        # shape [batch_size, k, vocab_size]
        recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)

        return recovered_probs

    def _multinomial(
        probs: torch.Tensor,
        residual_rand: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_samples = 1
        if residual_rand is not None:
            # Use provided uniform random numbers
            eps = torch.finfo(probs.dtype).eps
            clamped_uniform = torch.clamp(residual_rand, min=eps, max=1.0 - eps)
            q = -torch.log(clamped_uniform)
        else:
            # Generate random exponential numbers
            q = torch.empty_like(probs)
            q.exponential_(1.0)
        return probs.div_(q).argmax(dim=1).view(-1, num_samples)

    batch_size, k, vocab_size = draft_probs.shape
    # shape [batch_size, k]
    first_rejected_token_idx = _get_first_rejected_token_idx(
        target_probs, draft_probs, draft_token_ids
    )

    recovered_probs = _get_recovered_probs(target_probs, draft_probs).reshape(
        batch_size * k, vocab_size
    )

    # NOTE: the recovered_probs are overwritten by this method.
    recovered_token_ids = _multinomial(
        recovered_probs,
        residual_rand,
    ).reshape(batch_size, k)

    return first_rejected_token_idx, recovered_token_ids


@pytest.fixture(scope="module")
def compiled_rejection_sampler_with_residuals(
    session: InferenceSession,
) -> Model:
    """Compile rejection_sampler_with_residuals once for the module."""
    device = session.devices[0]
    graph = rejection_sampler_with_residuals(
        device=DeviceRef.from_device(device), debug=True
    )
    return session.load(graph)


def test_rejection_sampler_with_residuals(
    session: InferenceSession, compiled_rejection_sampler_with_residuals: Model
) -> None:
    batch_size = 3
    num_steps = 4
    vocab_size = 5
    torch.manual_seed(0)

    # num_steps +1 for bonus token
    target_logits = 3 * torch.randn(batch_size, num_steps + 1, vocab_size)

    draft_logits = target_logits[:, :-1] + 0.7 * torch.randn(
        batch_size, num_steps, vocab_size
    )
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)
    draft_token_ids = torch.argmax(draft_probs, dim=-1)

    # Generate controlled uniform random numbers for both rejection and multinomial sampling
    rejection_rand = torch.rand(batch_size, num_steps)
    residual_rand = torch.rand(batch_size * num_steps, vocab_size)

    first_rejected_token_idx, recovered_token_ids = rejection_sampler_reference(
        target_probs[:, :-1],
        draft_probs,
        draft_token_ids,
        rejection_rand,
        residual_rand,
    )

    device = session.devices[0]
    draft_logits_for_sampled_tokens = torch.gather(
        draft_logits, dim=-1, index=draft_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    target_logit_offsets = np.arange(
        0, (batch_size + 1) * (num_steps + 1), num_steps + 1
    )
    target_logits_tensor = target_logits.reshape(
        batch_size * (num_steps + 1), vocab_size
    )

    draft_logits_tensor = draft_logits.permute(1, 0, 2).contiguous()

    first_rejected_token, recovered_tokens, _bonus_tokens = (
        compiled_rejection_sampler_with_residuals(
            Buffer.from_dlpack(draft_token_ids).to(device),
            Buffer.from_dlpack(draft_logits_for_sampled_tokens).to(device),
            Buffer.from_dlpack(target_logits_tensor).to(device),
            Buffer.from_dlpack(target_logit_offsets).to(device),
            Buffer.from_dlpack(draft_logits_tensor).to(device),
            Buffer.from_numpy(np.array([0], dtype=np.uint64)).to(device),
            Buffer.from_dlpack(rejection_rand).to(device),
            Buffer.from_dlpack(
                residual_rand.reshape(batch_size, num_steps, vocab_size)
            ).to(device),
        )
    )

    # Now we can compare the results deterministically
    assert isinstance(first_rejected_token, Buffer)
    assert isinstance(recovered_tokens, Buffer)

    first_rejected_token_np = first_rejected_token.to_numpy()
    recovered_tokens_np = recovered_tokens.to_numpy()

    # Compare first rejected token indices
    np.testing.assert_array_equal(
        first_rejected_token_np,
        first_rejected_token_idx.numpy(),
        err_msg="First rejected token indices should match",
    )

    # Compare recovered token IDs
    np.testing.assert_array_equal(
        recovered_tokens_np,
        recovered_token_ids.numpy(),
        err_msg="Recovered token IDs should match",
    )
