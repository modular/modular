# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle
from typing import Union

import numpy as np
import pytest
from max.interfaces import (
    GenerationStatus,
    SamplingParams,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.pipelines.core import TextAndVisionContext, TextContext, TTSContext


def test_context__get_min_token_logit_mask() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4},
        sampling_params=SamplingParams(min_new_tokens=3),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4]]

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4]]

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == [[0, 4]]
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_multiple_eos_token_ids() -> (
    None
):
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        sampling_params=SamplingParams(min_new_tokens=3),
        eos_token_ids={4, 5},
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_multiple_eos_token_ids_multistep() -> (
    None
):
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        sampling_params=SamplingParams(min_new_tokens=3),
        eos_token_ids={4, 5},
    )
    vocab_mask = context.get_min_token_logit_mask(4)
    assert len(vocab_mask) == 4
    assert vocab_mask[0].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[1].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[2].tolist() == [[0, 4], [0, 5]]
    assert vocab_mask[3].tolist() == []

    context.update(1)
    context.update(1)
    context.update(1)
    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []


def test_context__get_min_token_logit_mask_with_no_eos_token_ids() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        sampling_params=SamplingParams(min_new_tokens=3),
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == []
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__get_min_token_logit_mask_with_no_min_new_tokens() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4, 5},
    )
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(1)
    vocab_mask = context.get_min_token_logit_mask(1)
    assert len(vocab_mask) == 1
    assert vocab_mask[0].tolist() == []

    context.update(2)
    vocab_mask = context.get_min_token_logit_mask(3)
    assert len(vocab_mask) == 3
    assert vocab_mask[0].tolist() == []
    assert vocab_mask[1].tolist() == []
    assert vocab_mask[2].tolist() == []


def test_context__eos() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
        eos_token_ids={4},
    )
    assert context.eos_token_ids == {4}
    assert context.is_initial_prompt == True
    context.update(4)
    assert context.is_initial_prompt == False
    assert context.current_length == 5
    assert context.status == GenerationStatus.END_OF_SEQUENCE


def test_context__max_length() -> None:
    context = TextContext(
        max_length=6,
        tokens=np.array([0, 1, 2, 3]),
    )
    for i in range(2):
        assert context.status == GenerationStatus.ACTIVE
        context.update(i)
    assert context.status == GenerationStatus.MAXIMUM_LENGTH


def test_context__current_length() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

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


def test_context__seq_len() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    assert context.active_length == 4
    context.update(4)
    assert context.active_length == 1
    for i in range(5):
        context.update(5 + i)
    assert context.active_length == 1


def test_context__bump_token_indices() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

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


def test_context__update_beyond_chunk_size() -> None:
    # This check evaluates whether we can update this array.
    # However, behaviour with max serve for updating, is slightly
    # different than behaviour off the server, as the text context
    # moves between the api worker & server worker.
    # Before making changes to resize behaviour, ensure you
    # test with the server, not just the `generate` entrypoint.
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # 128, is the CHUNK_SIZE defined in context
    for i in range(128):
        context.update(i)


def test_context__reset() -> None:
    context = TextContext(
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )
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


def test_context_sampling_params_integration() -> None:
    """Tests that TextContext properly stores and maintains SamplingParams."""
    custom_params = SamplingParams(
        top_k=25,
        temperature=0.7,
        frequency_penalty=0.4,
        presence_penalty=0.2,
        repetition_penalty=1.15,
    )

    context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        sampling_params=custom_params,
    )

    # Verify the sampling params persist through context operations
    context.update(5)
    assert context.sampling_params is custom_params
    assert context.sampling_params.top_k == 25

    context.reset()
    assert context.sampling_params is custom_params
    assert context.sampling_params.temperature == 0.7
    assert context.sampling_params.temperature == 0.7


def test_context_sampling_params_stop() -> None:
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        max_length=50,
        tokens=np.array([0]),
        eos_sequences=[[1, 2]],
        sampling_params=custom_params,
    )

    context.update(1)
    context.update(2)
    print(context.generated_tokens)
    assert context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 2]))

    context = TextContext(
        max_length=50,
        tokens=np.array([0]),
        eos_sequences=[[2], [3, 1]],
        sampling_params=custom_params,
    )
    context.update(1)
    context.update(3)

    assert not context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 3]))


def test_context_sampling_params_eos_token_ids() -> None:
    """Tests that TextContext can stop on user-defined sequences."""
    custom_params = SamplingParams(stop=["This is a test"])

    context = TextContext(
        max_length=50,
        tokens=np.array([0]),
        eos_token_ids=set([5, 4, 2]),
        sampling_params=custom_params,
    )
    context.update(1)
    context.update(2)

    assert context.is_done
    assert np.array_equal(context.generated_tokens, np.array([1, 2]))

    context = TextContext(
        max_length=50,
        tokens=np.array([0]),
        eos_token_ids=set([5, 4, 2]),
        sampling_params=custom_params,
    )
    context.update(3)
    context.update(6)

    assert not context.is_done
    assert np.array_equal(context.generated_tokens, np.array([3, 6]))


def test_context_serializable() -> None:
    # Test that we can encode a sample TextContext with Pickle
    original_context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextContext)
    assert pickle_decoded == original_context

    # Test that we can encode a sample TextContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        Union[TextContext, TextAndVisionContext]
    )
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded == original_context


def test_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, TextContext) with Pickle
    original_context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded == original_tuple

    # Test that we can encode a tuple of (str, TextContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        tuple[str, Union[TextContext, TextAndVisionContext]]
    )
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded == original_tuple


def test_text_and_vision_context_serializable() -> None:
    # Test that we can encode a sample TextAndVisionContext with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=(np.array([10, 11, 12, 13, 14]),),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextAndVisionContext)
    assert pickle_decoded == original_context

    # Test that we can encode a sample TextAndVisionContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        Union[TextContext, TextAndVisionContext]
    )
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded == original_context


def test_text_and_vision_context_serializable_empty_pixel_values() -> None:
    # Test that we can encode a sample TextAndVisionContext with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=tuple(),
    )

    pickle_encoded = pickle.dumps(original_context)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert isinstance(pickle_decoded, TextAndVisionContext)
    assert pickle_decoded == original_context

    # Test that we can encode a sample TextAndVisionContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        Union[TextContext, TextAndVisionContext]
    )
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded == original_context


def test_text_and_vision_context_tuple_serializable() -> None:
    # Test that we can encode a tuple of (str, TextAndVisionContext) with Pickle
    original_context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=(np.array([10, 11, 12, 13, 14]),),
    )
    original_tuple = ("test_key", original_context)

    pickle_encoded = pickle.dumps(original_tuple)
    pickle_decoded = pickle.loads(pickle_encoded)

    assert pickle_decoded == original_tuple

    # Test that we can encode a tuple of (str, TextAndVisionContext) with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(
        tuple[str, Union[TextContext, TextAndVisionContext]]
    )
    msgpack_encoded = serialize(original_tuple)
    msgpack_decoded = deserialize(msgpack_encoded)

    assert msgpack_decoded == original_tuple


def test_tts_context_msgpack_serialization_and_speech_tokens() -> None:
    """Tests that TTSContext can be serialized/deserialized with msgpack and that _speech_tokens can be written to after deserialization."""
    # Create a TTSContext with some audio prompt tokens
    audio_prompt_tokens = np.array([100, 101, 102, 103], dtype=np.int32)
    original_context = TTSContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        audio_prompt_tokens=audio_prompt_tokens,
        sampling_params=SamplingParams(temperature=0.8),
    )

    # Add some initial speech tokens to the context
    initial_speech_tokens = np.array([200, 201, 202], dtype=np.int32)
    original_context.update_speech_tokens(initial_speech_tokens)

    # Verify initial state
    assert np.array_equal(
        original_context.audio_prompt_tokens, audio_prompt_tokens
    )
    assert np.array_equal(original_context.speech_tokens, initial_speech_tokens)
    assert original_context._speech_token_end_idx == 3
    assert original_context.block_counter == 1

    # Test that we can encode TTSContext with MsgPack
    serialize = msgpack_numpy_encoder()
    deserialize = msgpack_numpy_decoder(TTSContext)
    msgpack_encoded = serialize(original_context)
    msgpack_decoded = deserialize(msgpack_encoded)

    # Verify the deserialized context matches the original
    assert isinstance(msgpack_decoded, TTSContext)
    assert msgpack_decoded == original_context
    assert np.array_equal(
        msgpack_decoded.audio_prompt_tokens, audio_prompt_tokens
    )
    assert np.array_equal(msgpack_decoded.speech_tokens, initial_speech_tokens)
    assert msgpack_decoded._speech_token_end_idx == 3
    assert msgpack_decoded.block_counter == 1

    # Test writing to the _speech_tokens array after deserialization
    new_speech_tokens = np.array([300, 301, 302, 303], dtype=np.int32)
    msgpack_decoded.update_speech_tokens(new_speech_tokens)

    # Verify that the new speech tokens were added correctly
    expected_combined_tokens = np.concatenate(
        [initial_speech_tokens, new_speech_tokens]
    )
    assert np.array_equal(
        msgpack_decoded.speech_tokens, expected_combined_tokens
    )
    assert msgpack_decoded._speech_token_end_idx == 7
    assert msgpack_decoded.block_counter == 2

    # Verify that the original context was not affected
    assert np.array_equal(original_context.speech_tokens, initial_speech_tokens)
    assert original_context._speech_token_end_idx == 3
    assert original_context.block_counter == 1


def test_vision_context_reset() -> None:
    context = TextAndVisionContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
        pixel_values=(np.array([10, 11, 12, 13, 14]),),
    )
    assert len(context.pixel_values) == 1
    assert context.pixel_values[0].tolist() == [10, 11, 12, 13, 14]
    assert context.start_idx == 0
    assert context.active_length == 5
    assert context.needs_vision_encoding is True

    # The pixel values should remain set after update, but needs_vision_encoding should be False.
    context.update(5)
    assert len(context.pixel_values) == 1
    assert context.pixel_values[0].tolist() == [10, 11, 12, 13, 14]
    assert context.needs_vision_encoding is False
    assert context.start_idx == 5
    assert context.active_length == 1

    # The pixel values should be restored after reset.
    context.reset()
    assert len(context.pixel_values) == 1
    assert context.pixel_values[0].tolist() == [10, 11, 12, 13, 14]
    assert context.start_idx == 0
    assert context.active_length == 6
    assert context.needs_vision_encoding is True
