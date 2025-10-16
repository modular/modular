# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize(
    ("vocab_size", "check_frequency"), ((64, True), (152064, False))
)
def test_gumbel_sampling(
    session: InferenceSession, vocab_size: int, check_frequency: bool
) -> None:
    """Test that gumbel sampling produces the correct results."""
    device = session.devices[0]
    device_ref = DeviceRef.from_device(device)

    # Test parameters
    temp = 1.0
    batch_size = 128
    num_trials = 100

    # TODO(E2EOPT-315) -- Currently, for a request requires top_k = -1, MAX automatically rewrites top_k to 255.
    # Hence, any request requires top_k >= 255 would be routed to gumbel sampling.
    top_k = 255

    # set numpy seed
    np.random.seed(0)

    logits_np = np.random.randn(vocab_size).astype(np.float32)
    # broadcast logits to batch size
    logits_np = np.tile(logits_np, (batch_size, 1))

    # Create graph with logits and seed as inputs
    logits_type = TensorType(
        DType.float32, ["batch_size", vocab_size], device=device_ref
    )
    seed_type = TensorType(DType.uint64, ["batch_size"], device=device_ref)
    top_p_type = TensorType(DType.float32, ["batch_size"], device=device_ref)
    top_k_type = TensorType(DType.int64, ["batch_size"], device=device_ref)
    max_k_type = TensorType(DType.int64, [], device=DeviceRef.CPU())
    temperature_type = TensorType(
        DType.float32, ["batch_size"], device=device_ref
    )

    def create_sampling_graph():  # noqa: ANN202
        with Graph(
            "gumbel_sampling",
            input_types=(
                logits_type,
                seed_type,
                top_p_type,
                top_k_type,
                max_k_type,
                temperature_type,
            ),
        ) as graph:
            logits_input = graph.inputs[0].tensor
            seed_input = graph.inputs[1].tensor
            top_p_input = graph.inputs[2].tensor
            top_k_input = graph.inputs[3].tensor
            max_k_input = graph.inputs[4].tensor
            temperature_input = graph.inputs[5].tensor

            # We need to manually create the custom op since topk_fused_sampling
            # doesn't accept seed as tensor
            sampled_tokens = ops.custom(
                "sampler.fused_token_sampling",
                device=device_ref,
                values=[
                    top_k_input,
                    max_k_input,
                    temperature_input,
                    top_p_input,
                    seed_input,
                    logits_input,
                ],
                out_types=[
                    TensorType(
                        dtype=DType.int64,
                        shape=[logits_input.shape[0], 1],
                        device=device_ref,
                    )
                ],
            )[0].tensor

            graph.output(sampled_tokens)
        return graph

    graph = create_sampling_graph()
    sampler = session.load(graph)

    logits_tensor = Tensor.from_dlpack(logits_np).to(device)
    sampled_tokens = []

    temperature = Tensor.from_numpy(
        np.array([temp] * batch_size, dtype=np.float32)
    ).to(device)
    top_k_tensor = Tensor.from_numpy(
        np.array([top_k] * batch_size, dtype=np.int64)
    ).to(device)
    max_k = Tensor.from_numpy(np.array(top_k, dtype=np.int64))
    top_p_tensor = Tensor.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    for seed in range(num_trials):
        seed_start = seed * batch_size
        seed_end = seed_start + batch_size
        seed_tensor = Tensor.from_dlpack(
            np.arange(seed_start, seed_end, dtype=np.uint64)
        ).to(device)
        tokens = sampler(
            logits_tensor,
            seed_tensor,
            top_p_tensor,
            top_k_tensor,
            max_k,
            temperature,
        )[0]
        assert isinstance(tokens, Tensor)
        token_idxs = tokens.to_numpy()
        sampled_tokens.append(token_idxs)

    if check_frequency:
        # calculate softmax of logits
        single_token_np = logits_np[0, :]
        softmax_np = np.exp(single_token_np) / np.sum(np.exp(single_token_np))

        # count frequency of each token in sampled tokens
        all_tokens = np.concatenate(sampled_tokens)
        unique_tokens, counts = np.unique(all_tokens, return_counts=True)

        # check up to 10 most frequent tokens
        most_frequent_idx = np.argsort(counts)[-10:]

        token = unique_tokens[most_frequent_idx]
        sample_freq = counts[most_frequent_idx] / (batch_size * num_trials)
        ref_freq = softmax_np[token]

        np.testing.assert_allclose(sample_freq, ref_freq, atol=0.01, rtol=0.05)
