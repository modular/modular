# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle

import numpy as np
import pytest
from max.pipelines.core import (
    SamplingParams,
    TextContext,
    TextGenerationStatus,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)


def test_context__eos():
    context = TextContext(
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4},
    )
    context.assign_to_cache(cache_seq_id=0)
    assert context.eos_token_ids == {4}
    assert context.is_initial_prompt == True
    context.update(4)
    assert context.is_initial_prompt == False
    assert context.current_length == 5
    assert context.status == TextGenerationStatus.END_OF_SEQUENCE


def test_context__max_length():
    context = TextContext(
        prompt="this is a test prompt",
        max_length=6,
        tokens=np.array([0, 1, 2, 3]),
    )
    context.assign_to_cache(cache_seq_id=0)
    for i in range(2):
        assert context.status == TextGenerationStatus.ACTIVE
        context.update(i)
    assert context.status == TextGenerationStatus.MAXIMUM_LENGTH


def test_context__current_length():
    context = TextContext(
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
    context.assign_to_cache(0)

    assert context.current_length == 4
    assert context.is_initial_prompt == True

    context.update(4)
    assert context.is_initial_prompt == False
    assert context.current_length == 5

    # Currently, there are 5 tokens, we are saying
    # here is the next one, and we've generated 3 tokens
    # including that one, so increment the current length
    # accordingly.
    for i in range(3):
        context.update(5 + i)

    assert context.current_length == 8


def test_context__seq_len():
    context = TextContext(
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
    context.assign_to_cache(0)

    assert context.active_length == 4
    context.update(4)
    assert context.active_length == 1
    for i in range(5):
        context.update(5 + i)
    assert context.active_length == 1


def test_context__bump_token_indices():
    context = TextContext(
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
    context.assign_to_cache(0)

    # Can't trim more tokens than the context has.
    with pytest.raises(ValueError):
        context.bump_token_indices(start_idx=999)

    # Trimming 0 tokens does nothing.
    context.bump_token_indices(start_idx=0)
    assert (context.next_tokens == np.array([0, 1, 2, 3])).all()
    assert context.active_length == 4
    assert context.current_length == 4

    # Trimming 2 tokens should remove the first 2 tokens of prompt.
    context.bump_token_indices(start_idx=2)
    assert (context.next_tokens == np.array([2, 3])).all()
    assert context.active_length == 2
    assert context.current_length == 4  # does not change

    # Can't trim prompt to 0 tokens.
    with pytest.raises(ValueError):
        context.bump_token_indices(start_idx=2)


def test_context__update_beyond_chunk_size():
    # This check evaluates whether we can update this array.
    # However, behaviour with max serve for updating, is slightly
    # different than behaviour off the server, as the text context
    # moves between the api worker & server worker.
    # Before making changes to resize behaviour, ensure you
    # test with the server, not just the `generate` entrypoint.
    context = TextContext(
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
    context.assign_to_cache(0)

    # 128, is the CHUNK_SIZE defined in context
    for i in range(128):
        context.update(i)


def test_context__reset():
    context = TextContext(
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
    context.assign_to_cache(0)
    assert context.active_length == 4
    assert context.next_tokens.tolist() == [0, 1, 2, 3]
    context.update(4)
    assert context.active_length == 1
    assert context.next_tokens.tolist() == [4]
    context.reset()
    assert context.active_length == 5
    assert context.next_tokens.tolist() == [0, 1, 2, 3, 4]
    context.update(5)
    assert context.active_length == 1
    assert context.next_tokens.tolist() == [5]


def test_context_sampling_params_integration():
    """Tests that TextContext properly stores and maintains SamplingParams."""
    custom_params = SamplingParams(
        top_k=25,
        temperature=0.7,
        frequency_penalty=0.4,
        presence_penalty=0.2,
        repetition_penalty=1.15,
        enable_structured_output=True,
        enable_variable_logits=False,
        do_penalties=True,
    )

    context = TextContext(
        prompt="sampling params test prompt",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        sampling_params=custom_params,
    )
    context.assign_to_cache(0)

    # Verify the sampling params persist through context operations
    context.update(5)
    assert context.sampling_params is custom_params
    assert context.sampling_params.top_k == 25

    context.reset()
    assert context.sampling_params is custom_params
    assert context.sampling_params.temperature == 0.7
    assert context.sampling_params.temperature == 0.7


def test_context_sampling_params_stop():
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        prompt="This is a test prompt",
        max_length=50,
        tokens=np.array([0]),
        eos_sequences=[[1, 2]],
        sampling_params=custom_params,
    )
    context.assign_to_cache(0)

    context.update(1)
    context.update(2)
    print(context.generated_tokens)
    assert context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 2]))

    context = TextContext(
        prompt="This is a test prompt",
        max_length=50,
        tokens=np.array([0]),
        eos_sequences=[[2], [3, 1]],
        sampling_params=custom_params,
    )
    context.assign_to_cache(0)
    context.update(1)
    context.update(3)

    assert not context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 3]))


def test_context_sampling_params_eos_token_ids():
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        prompt="This is a test prompt",
        max_length=50,
        tokens=np.array([0]),
        eos_token_ids=set([5, 4, 2]),
        sampling_params=custom_params,
    )
    context.assign_to_cache(0)
    context.update(1)
    context.update(2)

    assert context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 2]))

    context = TextContext(
        prompt="This is a test prompt",
        max_length=50,
        tokens=np.array([0]),
        eos_token_ids=set([5, 4, 2]),
        sampling_params=custom_params,
    )
    context.assign_to_cache(0)
    context.update(3)
    context.update(6)

    assert not context.is_done
    assert np.array_equal(context.generated_tokens, np.array([3, 6]))


def test_context_serializable():
    # Test that we can encode a sample TextContext with Pickle
    original_context = TextContext(
        prompt="sampling params test prompt",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextContext)
    assert pickle_decoded == original_context

    # Test that we can encode a sample TextContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TextContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded == original_context
